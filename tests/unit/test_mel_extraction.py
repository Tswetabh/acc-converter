"""
tests.unit.test_mel_extraction
==============================
Unit tests for mel spectrogram extraction (src/accent_converter/audio/mel.py).
"""

import numpy as np
import pytest
import torch

from accent_converter.audio.mel import MEL_CONFIG, extract_mel_spectrogram


class TestMelExtraction:
    """Unit tests for extract_mel_spectrogram."""

    def test_mel_config_keys_and_values(self):
        assert MEL_CONFIG["sample_rate"] == 16000
        assert MEL_CONFIG["n_mels"] == 80
        assert MEL_CONFIG["hop_length"] == 256
        assert MEL_CONFIG["win_length"] == 1024
        assert MEL_CONFIG["n_fft"] == 1024

    def test_extract_mel_from_numpy(self):
        # 1 second of audio at 16 kHz = 16000 samples
        audio = np.random.randn(16000).astype(np.float32)
        mel = extract_mel_spectrogram(audio)

        assert isinstance(mel, torch.Tensor)
        assert mel.ndim == 2
        assert mel.shape[0] == 80
        # For 16000 samples with hop=256, frames = ceil(16000/256) ≈ 63
        assert mel.shape[1] == 63
        assert mel.dtype == torch.float32

    def test_extract_mel_from_torch_tensor(self):
        audio = torch.randn(8000, dtype=torch.float32)
        mel = extract_mel_spectrogram(audio)

        assert isinstance(mel, torch.Tensor)
        assert mel.ndim == 2
        assert mel.shape[0] == 80
        assert mel.shape[1] > 0

    def test_extract_mel_squeeze_singleton_dims(self):
        # (1, 16000) or (16000, 1)
        audio_2d = torch.randn(1, 16000)
        mel = extract_mel_spectrogram(audio_2d)
        assert mel.shape[0] == 80
        assert mel.shape[1] == 63

    def test_extract_mel_numpy_float64_auto_converts(self):
        audio = np.random.randn(8000).astype(np.float64)
        mel = extract_mel_spectrogram(audio)
        assert mel.dtype == torch.float32

    def test_extract_mel_empty_audio(self):
        empty_np = np.empty(0, dtype=np.float32)
        mel = extract_mel_spectrogram(empty_np)
        assert mel.shape == (80, 0)

        empty_torch = torch.empty(0, dtype=torch.float32)
        mel_t = extract_mel_spectrogram(empty_torch)
        assert mel_t.shape == (80, 0)

    def test_extract_mel_invalid_ndim(self):
        bad_np = np.random.randn(2, 4, 16000).astype(np.float32)
        with pytest.raises(ValueError, match="audio must be 1D"):
            extract_mel_spectrogram(bad_np)

        bad_torch = torch.randn(2, 4, 16000)
        with pytest.raises(ValueError, match="audio must be 1D"):
            extract_mel_spectrogram(bad_torch)

    def test_extract_mel_invalid_sample_rate(self):
        audio = np.random.randn(16000).astype(np.float32)
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            extract_mel_spectrogram(audio, sample_rate=0)

    def test_extract_mel_invalid_type(self):
        with pytest.raises(TypeError, match="np.ndarray or torch.Tensor"):
            extract_mel_spectrogram("not_audio")  # type: ignore

    def test_extract_mel_numpy_and_torch_identical(self):
        np_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000, endpoint=False)).astype(np.float32)
        torch_audio = torch.from_numpy(np_audio.copy())

        mel_np = extract_mel_spectrogram(np_audio)
        mel_torch = extract_mel_spectrogram(torch_audio)

        torch.testing.assert_close(mel_np, mel_torch, rtol=1e-5, atol=1e-5)
