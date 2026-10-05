"""
audio.mel
=========
Mel spectrogram extraction matching SpeechBrain's multi-speaker 16 kHz HiFi-GAN
(``speechbrain/tts-hifigan-libritts-16kHz``).

Specifications
--------------
- Sample rate: 16,000 Hz
- FFT size: 1024
- Window length: 1024
- Hop length: 256 (62.5 Hz frame rate)
- Mel bands: 80
- Frequency range: 0.0 – 8000.0 Hz
- Power: 1 (magnitude)
- Mel scale: Slaney
- Normalization: Slaney area normalization
- Dynamic range compression: True (log scale)
"""

from __future__ import annotations

from typing import Any
import numpy as np
import torch

MEL_CONFIG: dict[str, Any] = {
    "sample_rate": 16000,
    "n_fft": 1024,
    "win_length": 1024,
    "hop_length": 256,
    "n_mels": 80,
    "f_min": 0.0,
    "f_max": 8000.0,
    "power": 1,
    "normalized": False,
    "min_max_energy_norm": True,
    "norm": "slaney",
    "mel_scale": "slaney",
    "compression": True,
}


def extract_mel_spectrogram(
    audio: np.ndarray | torch.Tensor,
    sample_rate: int = 16000,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Extract an 80-channel log-mel spectrogram matching SpeechBrain HiFi-GAN.

    Parameters
    ----------
    audio:
        1D waveform as a float32 NumPy array or torch.Tensor.
    sample_rate:
        Audio sample rate. Must be 16000 for the native pipeline.
    device:
        Torch device to perform extraction on (default: audio tensor's device,
        or CPU if NumPy array).

    Returns
    -------
    torch.Tensor
        Spectrogram tensor with shape ``(80, T_mel)`` of type float32.

    Raises
    ------
    ValueError:
        If *audio* is not 1D or *sample_rate* is invalid.
    TypeError:
        If *audio* is neither a NumPy array nor a torch.Tensor.
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")

    if isinstance(audio, np.ndarray):
        if audio.ndim != 1:
            raise ValueError(f"audio must be 1D, got ndim={audio.ndim} with shape {audio.shape}")
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        tensor = torch.from_numpy(audio)
        if device is not None:
            tensor = tensor.to(device)
    elif isinstance(audio, torch.Tensor):
        if audio.ndim == 2 and (audio.shape[0] == 1 or audio.shape[1] == 1):
            audio = audio.squeeze()
        if audio.ndim != 1:
            raise ValueError(f"audio must be 1D, got ndim={audio.ndim} with shape {tuple(audio.shape)}")
        if audio.dtype != torch.float32:
            audio = audio.to(torch.float32)
        tensor = audio if device is None else audio.to(device)
    else:
        raise TypeError(f"audio must be np.ndarray or torch.Tensor, got {type(audio).__name__}")

    if tensor.numel() == 0:
        return torch.zeros((MEL_CONFIG["n_mels"], 0), dtype=torch.float32, device=tensor.device)

    from speechbrain.lobes.models.FastSpeech2 import mel_spectrogram

    mel, _ = mel_spectrogram(
        sample_rate=sample_rate,
        hop_length=MEL_CONFIG["hop_length"],
        win_length=MEL_CONFIG["win_length"],
        n_fft=MEL_CONFIG["n_fft"],
        n_mels=MEL_CONFIG["n_mels"],
        f_min=MEL_CONFIG["f_min"],
        f_max=MEL_CONFIG["f_max"],
        power=MEL_CONFIG["power"],
        normalized=MEL_CONFIG["normalized"],
        min_max_energy_norm=MEL_CONFIG["min_max_energy_norm"],
        norm=MEL_CONFIG["norm"],
        mel_scale=MEL_CONFIG["mel_scale"],
        compression=MEL_CONFIG["compression"],
        audio=tensor,
    )

    return mel
