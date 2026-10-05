"""
tests/unit/test_audio_resampler.py
====================================
Unit tests for accent_converter.audio.resampler

Covers:
- passthrough at 16 kHz (no computation)
- 44.1 kHz → 16 kHz
- 48 kHz → 16 kHz
- 32 kHz → 16 kHz
- 8 kHz → 16 kHz (upsampling)
- output length correctness
- output dtype preservation
- multi-channel input raises ValueError
- invalid sample rate raises ValueError
- DC offset preserved after resampling
"""

import math

import numpy as np
import pytest

from accent_converter.audio.resampler import resample, TARGET_SAMPLE_RATE


# ===========================================================================
# Helpers
# ===========================================================================


def _sine(freq: float, sr: int, duration: float = 0.2) -> np.ndarray:
    """1-D float32 sine wave."""
    t = np.arange(int(sr * duration), dtype=np.float64) / sr
    return np.sin(2.0 * np.pi * freq * t).astype(np.float32)


def _expected_length(orig_sr: int, duration_s: float, target_sr: int = 16_000) -> int:
    """Expected output sample count after resampling."""
    return round(int(orig_sr * duration_s) * target_sr / orig_sr)


# ===========================================================================
# Passthrough
# ===========================================================================


class TestPassthrough:
    def test_16k_is_passthrough(self):
        """Input already at 16 kHz → returned copy, no resampling."""
        audio = _sine(440.0, 16_000)
        result = resample(audio, orig_sr=16_000)
        assert result is not audio        # must be a copy
        np.testing.assert_array_equal(result, audio)

    def test_passthrough_preserves_dtype(self):
        audio = _sine(440.0, 16_000).astype(np.float64)
        result = resample(audio, orig_sr=16_000, target_sr=16_000)
        assert result.dtype == np.float64


# ===========================================================================
# Sample-rate conversion (downsampling)
# ===========================================================================


class TestDownsampling:
    """Test common input rates → 16 kHz."""

    DURATION = 0.5   # seconds — long enough for accurate length check

    @pytest.mark.parametrize("orig_sr", [44_100, 48_000, 32_000, 22_050])
    def test_output_length(self, orig_sr: int):
        """Output length matches expected rational resample length."""
        audio = _sine(440.0, orig_sr, self.DURATION)
        result = resample(audio, orig_sr=orig_sr)
        expected = _expected_length(orig_sr, self.DURATION)
        # Allow ±1 sample for rounding differences in polyphase filter
        assert abs(len(result) - expected) <= 1, (
            f"orig_sr={orig_sr}: expected ~{expected} samples, got {len(result)}"
        )

    @pytest.mark.parametrize("orig_sr", [44_100, 48_000, 32_000])
    def test_output_is_1d(self, orig_sr: int):
        audio = _sine(440.0, orig_sr)
        result = resample(audio, orig_sr=orig_sr)
        assert result.ndim == 1

    def test_44100_to_16000(self):
        """Spot-check: 44 100 → 16 000 length ratio ≈ 16/44.1."""
        audio = _sine(440.0, 44_100, self.DURATION)
        result = resample(audio, orig_sr=44_100)
        ratio = len(result) / len(audio)
        expected_ratio = 16_000 / 44_100
        assert abs(ratio - expected_ratio) < 0.01

    def test_48000_to_16000(self):
        """48 kHz → 16 kHz: exact 1/3 ratio."""
        audio = _sine(440.0, 48_000, self.DURATION)
        result = resample(audio, orig_sr=48_000)
        expected_len = int(len(audio) / 3)
        assert abs(len(result) - expected_len) <= 1

    def test_32000_to_16000(self):
        """32 kHz → 16 kHz: exact 1/2 ratio."""
        audio = _sine(440.0, 32_000, self.DURATION)
        result = resample(audio, orig_sr=32_000)
        expected_len = len(audio) // 2
        assert abs(len(result) - expected_len) <= 1


# ===========================================================================
# Upsampling
# ===========================================================================


class TestUpsampling:
    def test_8000_to_16000_length(self):
        """8 kHz → 16 kHz: exact 2× ratio."""
        audio = _sine(300.0, 8_000, duration=0.5)
        result = resample(audio, orig_sr=8_000)
        expected_len = len(audio) * 2
        assert abs(len(result) - expected_len) <= 1


# ===========================================================================
# dtype preservation
# ===========================================================================


class TestDtypePreservation:
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_dtype_preserved(self, dtype):
        audio = _sine(440.0, 44_100).astype(dtype)
        result = resample(audio, orig_sr=44_100)
        assert result.dtype == dtype


# ===========================================================================
# Error handling
# ===========================================================================


class TestErrorHandling:
    def test_2d_array_raises(self):
        """Multi-channel (2-D) input raises ValueError."""
        stereo = np.zeros((2, 1000), dtype=np.float32)
        with pytest.raises(ValueError, match="1-D"):
            resample(stereo, orig_sr=44_100)

    def test_invalid_orig_sr_zero(self):
        with pytest.raises(ValueError, match="orig_sr"):
            resample(np.zeros(100, dtype=np.float32), orig_sr=0)

    def test_invalid_orig_sr_negative(self):
        with pytest.raises(ValueError, match="orig_sr"):
            resample(np.zeros(100, dtype=np.float32), orig_sr=-44100)

    def test_invalid_orig_sr_string(self):
        with pytest.raises((ValueError, TypeError)):
            resample(np.zeros(100, dtype=np.float32), orig_sr="44100")  # type: ignore[arg-type]

    def test_invalid_target_sr(self):
        with pytest.raises(ValueError, match="target_sr"):
            resample(np.zeros(100, dtype=np.float32), orig_sr=44_100, target_sr=0)


# ===========================================================================
# DC preservation
# ===========================================================================


class TestDCOffset:
    def test_dc_preserved(self):
        """A DC-offset signal should retain its offset after resampling."""
        audio = np.full(44_100, 0.5, dtype=np.float64)
        result = resample(audio, orig_sr=44_100)
        # Skip edge samples (filter transient); check interior
        interior = result[50:-50]
        np.testing.assert_allclose(interior, 0.5, atol=1e-4)


# ===========================================================================
# Constant export
# ===========================================================================


def test_target_sample_rate_constant():
    assert TARGET_SAMPLE_RATE == 16_000
