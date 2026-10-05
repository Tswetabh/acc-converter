"""
models.backends.accent.conformer
================================
Phase 6D: Conformer-based accent translator operating on HuBERT features.

Maps source-accent HuBERT features ``(T, 768)`` to target-accent HuBERT
features of the same length ``(T, 768)`` using a residual Conformer stack.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchaudio

from accent_converter.models.interfaces import AccentTranslator


class ConformerAccentNet(nn.Module):
    """Residual Conformer: out = x + proj(Conformer(in_proj(x)))."""

    def __init__(
        self,
        dim: int = 768,
        hidden: int = 256,
        heads: int = 4,
        ffn_dim: int = 1024,
        layers: int = 6,
        kernel: int = 31,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(dim, hidden)
        self.conformer = torchaudio.models.Conformer(
            input_dim=hidden,
            num_heads=heads,
            ffn_dim=ffn_dim,
            num_layers=layers,
            depthwise_conv_kernel_size=kernel,
            dropout=dropout,
        )
        self.out_proj = nn.Linear(hidden, dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h, _ = self.conformer(h, lengths)
        return x + self.out_proj(h)


class ConformerAccentTranslator(AccentTranslator):
    """Learned accent translator backend (Hindi-accent -> US-accent)."""

    BACKEND_NAME: str = "conformer"

    def __init__(self, device: str = "cpu", **model_kwargs) -> None:
        self._device = device
        self._model_kwargs = model_kwargs
        self._model: ConformerAccentNet | None = None

    def load(self, checkpoint_path: str) -> None:
        ckpt = torch.load(checkpoint_path, map_location=self._device, weights_only=False)
        kwargs = ckpt.get("model_kwargs", self._model_kwargs)
        self._model = ConformerAccentNet(**kwargs).to(self._device)
        self._model.load_state_dict(ckpt["model"])
        self._model.eval()

    @torch.no_grad()
    def translate(self, content_rep: np.ndarray, target_accent: str) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call load() before translate().")
        x = torch.from_numpy(content_rep).float().unsqueeze(0).to(self._device)
        lengths = torch.tensor([x.shape[1]], device=self._device)
        return self._model(x, lengths)[0].cpu().numpy()

    def reset(self) -> None:
        pass
