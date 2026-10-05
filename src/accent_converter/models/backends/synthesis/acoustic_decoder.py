"""
models.backends.synthesis.acoustic_decoder
==========================================
Neural Acoustic Decoder mapping HuBERT content features and ECAPA-TDNN speaker
embeddings to 80-channel log-mel spectrograms.

Architecture
------------
1. Speaker Conditioning: ECAPA embedding (192,) projected to content dimension
   and added to HuBERT features (T, 768).
2. Input Projection: Linear layer (768 -> hidden_dim).
3. Temporal Resampling: Linear interpolation from HuBERT frame rate (50 Hz)
   to Mel frame rate (62.5 Hz, hop=256 @ 16kHz).
4. Temporal Modeling: Stack of 1D Residual Convolution blocks with LayerNorm
   and GELU activations.
5. Output Mel Projection: Projects hidden representations to 80 mel channels.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Residual 1D Convolution block with LayerNorm and GELU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Tensor of shape ``(B, C, T)``

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(B, C, T)``
        """
        residual = x

        # Layer 1
        x = self.conv1(x)
        x = x.transpose(1, 2)  # (B, T, C) for LayerNorm
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # (B, C, T)

        # Layer 2
        x = self.conv2(x)
        x = x.transpose(1, 2)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)

        return x + residual


class AcousticDecoder(nn.Module):
    """Maps HuBERT continuous features + ECAPA speaker embedding to 80-bin mel spectrogram.

    Parameters
    ----------
    content_dim:
        Dimension of HuBERT content features (default 768).
    speaker_dim:
        Dimension of ECAPA speaker embedding (default 192).
    mel_dim:
        Number of output mel frequency bins (default 80).
    hidden_dim:
        Intermediate channel dimension (default 512).
    n_conv_blocks:
        Number of residual convolution blocks (default 4).
    kernel_size:
        Kernel size for 1D convolutions (default 5).
    dropout:
        Dropout probability (default 0.1).
    """

    def __init__(
        self,
        content_dim: int = 768,
        speaker_dim: int = 192,
        mel_dim: int = 80,
        hidden_dim: int = 512,
        n_conv_blocks: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.content_dim = content_dim
        self.speaker_dim = speaker_dim
        self.mel_dim = mel_dim
        self.hidden_dim = hidden_dim

        # Speaker conditioning projection
        self.speaker_proj = nn.Linear(speaker_dim, content_dim)

        # Input feature projection
        self.input_proj = nn.Linear(content_dim, hidden_dim)

        # Residual convolution blocks
        self.conv_blocks = nn.ModuleList([
            ConvBlock(hidden_dim, hidden_dim, kernel_size=kernel_size, dropout=dropout)
            for _ in range(n_conv_blocks)
        ])

        # Final projection to mel channels
        self.mel_proj = nn.Linear(hidden_dim, mel_dim)

    def forward(
        self,
        hubert_features: torch.Tensor,
        speaker_embedding: torch.Tensor,
        target_len: int | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        hubert_features:
            Shape ``(B, T_hubert, 768)`` or ``(T_hubert, 768)``.
        speaker_embedding:
            Shape ``(B, 192)`` or ``(192,)``.
        target_len:
            Optional target number of mel frames. If None, automatically computed
            as ``round(T_hubert * 1.25)`` (50 Hz -> 62.5 Hz).

        Returns
        -------
        torch.Tensor
            Mel spectrogram with shape ``(B, 80, T_mel)``.
        """
        # Batch dimension normalization
        if hubert_features.ndim == 2:
            hubert_features = hubert_features.unsqueeze(0)  # (1, T, 768)
        if speaker_embedding.ndim == 1:
            speaker_embedding = speaker_embedding.unsqueeze(0)  # (1, 192)

        B, T_hubert, D = hubert_features.shape
        if D != self.content_dim:
            raise ValueError(
                f"Expected content_dim={self.content_dim}, got {D} in {hubert_features.shape}"
            )
        if speaker_embedding.shape[-1] != self.speaker_dim:
            raise ValueError(
                f"Expected speaker_dim={self.speaker_dim}, got {speaker_embedding.shape[-1]}"
            )

        if T_hubert == 0:
            return torch.zeros((B, self.mel_dim, 0), dtype=hubert_features.dtype, device=hubert_features.device)

        # 1. Condition on speaker
        spk = self.speaker_proj(speaker_embedding)  # (B, content_dim)
        x = hubert_features + spk.unsqueeze(1)      # (B, T_hubert, content_dim)

        # 2. Input projection to hidden_dim
        x = self.input_proj(x)                      # (B, T_hubert, hidden_dim)

        # 3. Resample temporal dimension from 50 Hz to 62.5 Hz (Mel frame rate)
        # Transpose to (B, hidden_dim, T_hubert) for 1D interpolation
        x = x.transpose(1, 2)
        if target_len is None:
            t_mel = max(1, int(round(T_hubert * 1.25)))
        else:
            t_mel = max(1, target_len)

        if t_mel != T_hubert:
            x = F.interpolate(x, size=t_mel, mode="linear", align_corners=False)

        # 4. Temporal modeling with residual Conv blocks
        for block in self.conv_blocks:
            x = block(x)

        # 5. Project to mel bins: (B, hidden_dim, T_mel) -> (B, T_mel, hidden_dim) -> (B, T_mel, mel_dim)
        x = x.transpose(1, 2)
        mel = self.mel_proj(x)                      # (B, T_mel, mel_dim)

        # HiFi-GAN expects (B, n_mels, T_mel)
        mel = mel.transpose(1, 2)                   # (B, mel_dim, T_mel)
        return mel
