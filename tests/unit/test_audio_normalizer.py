"""
tests/unit/test_audio_normalizer.py
====================================
Unit tests for accent_converter.audio.normalizer

Covers:
- mono conversion from stereo (channels-first and channels-last)
- mono passthrough (already mono)
- float32 conversion from int16
- float32 conversion from int32
- float32 conversion from uint8
- float32 passthrough (already float32)
- float64 -> float32 cast
- amplitude / peak normalisation
- silent frame handling (no division by zero)
- combined normalize() pipeline
"""

import numpy as np
import pytest

from accent_converter.audio.normalizer import (
    to_mono,
    to_float32,
    peak_normalize,
    normalize,
    CANONICAL_SAMPLE_RATE,
    CANONICAL_DTYPE,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _sine(freq: float = 440.0, sr: int = 16_000, duration: float = 0.1) -> np.ndarray:
    """Return a float32 sine wave of length sr*duration."""
    t = np.arange(int(sr * duration), dtype=np.float32) / sr
    return np.sin(2.0 * np.pi * freq * t, dtype=np.float32)


# ===========================================================================
# to_mono
# ===========================================================================


class TestToMono:
    def test_already_mono_1d(self):
        """1-D array is returned as-is (same object)."""
        audio = _sine()
        result = to_mono(audio)
        assert result is audio

    def test_stereo_channels_first(self):
        """(2, N) channels-first → averaged mono."""
        ch1 = np.ones(100, dtype=np.float32)
        ch2 = np.ones(100, dtype=np.float32) * 3.0
        stereo = np.stack([ch1, ch2], axis=0)   # shape (2, 100)
        mono = to_mono(stereo)
        assert mono.shape == (100,)
        np.testing.assert_allclose(mono, 2.0)

    def test_stereo_channels_last(self):
        """(N, 2) channels-last → averaged mono."""
        ch1 = np.ones(100, dtype=np.float32)
        ch2 = np.ones(100, dtype=np.float32) * 3.0
        stereo = np.stack([ch1, ch2], axis=1)   # shape (100, 2)
        mono = to_mono(stereo)
        assert mono.shape == (100,)
        np.testing.assert_allclose(mono, 2.0)

    def test_dtype_preserved_after_mono(self):
        """Output dtype matches input dtype."""
        stereo = np.array([[1, 2, 3], [3, 4, 5]], dtype=np.float32)
        mono = to_mono(stereo)
        assert mono.dtype == np.float32

    def test_3d_raises(self):
        """3-D input raises ValueError."""
        with pytest.raises(ValueError, match="1-D or 2-D"):
            to_mono(np.zeros((2, 100, 3)))


# ===========================================================================
# to_float32
# ===========================================================================


class TestToFloat32:
    def test_passthrough_float32(self):
        """float32 input is returned as the same object."""
        audio = _sine()
        assert audio.dtype == np.float32
        result = to_float32(audio)
        assert result is audio

    def test_float64_cast(self):
        """float64 is cast to float32 without rescaling."""
        audio = np.array([0.5, -0.5, 1.0], dtype=np.float64)
        result = to_float32(audio)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, [0.5, -0.5, 1.0], atol=1e-6)

    def test_int16_to_float32(self):
        """int16 is divided by 32 768 → range [-1, 1]."""
        audio = np.array([0, 32_767, -32_768], dtype=np.int16)
        result = to_float32(audio)
        assert result.dtype == np.float32
        # 32767 / 32768 ≈ 0.9999695
        assert abs(result[0]) < 1e-6
        assert abs(result[1] - 32_767 / 32_768) < 1e-5
        assert abs(result[2] - (-32_768 / 32_768)) < 1e-5
        # All values must be in [-1, 1]
        assert np.all(np.abs(result) <= 1.0 + 1e-6)

    def test_int32_to_float32(self):
        """int32 is divided by 2 147 483 648."""
        audio = np.array([0, 2_147_483_647, -2_147_483_648], dtype=np.int32)
        result = to_float32(audio)
        assert result.dtype == np.float32
        assert abs(result[0]) < 1e-6
        # Full-scale positive: ≈ 1.0
        assert 0.99 < result[1] <= 1.0 + 1e-6
        assert result[2] <= -0.99

    def test_uint8_to_float32(self):
        """uint8 midpoint (128) maps to 0, 0→-1, 255→≈1."""
        audio = np.array([128, 0, 255], dtype=np.uint8)
        result = to_float32(audio)
        assert result.dtype == np.float32
        assert abs(result[0]) < 1e-6        # 128 → 0
        assert abs(result[1] - (-1.0)) < 1e-6   # 0 → -1
        assert abs(result[2] - (127 / 128)) < 1e-5   # 255 → ~0.992

    def test_output_is_float32_dtype(self):
        """All input dtypes produce float32 output."""
        for dtype in (np.int16, np.int32, np.uint8, np.float64):
            arr = np.zeros(10, dtype=dtype)
            assert to_float32(arr).dtype == np.float32


# ===========================================================================
# peak_normalize
# ===========================================================================


class TestPeakNormalize:
    def test_peak_becomes_one(self):
        """After normalisation the absolute peak is exactly 1.0."""
        audio = np.array([0.2, -0.5, 0.3], dtype=np.float32)
        result = peak_normalize(audio)
        assert result.dtype == np.float32
        np.testing.assert_allclose(np.max(np.abs(result)), 1.0, atol=1e-6)

    def test_headroom_respected(self):
        """Peak equals headroom parameter."""
        audio = np.array([0.2, -0.5, 0.3], dtype=np.float32)
        result = peak_normalize(audio, headroom=0.9)
        np.testing.assert_allclose(np.max(np.abs(result)), 0.9, atol=1e-6)

    def test_silent_frame_unchanged(self):
        """All-zero input is returned unchanged (no division by zero)."""
        audio = np.zeros(100, dtype=np.float32)
        result = peak_normalize(audio)
        np.testing.assert_array_equal(result, audio)

    def test_near_silent_unchanged(self):
        """Sub-threshold amplitude (< 1e-8) is treated as silent."""
        audio = np.full(10, 1e-10, dtype=np.float32)
        result = peak_normalize(audio)
        np.testing.assert_allclose(result, audio, atol=1e-15)

    def test_amplitude_in_range(self):
        """Normalised sine wave stays in [-1, 1]."""
        audio = _sine() * 0.3
        result = peak_normalize(audio)
        assert np.all(np.abs(result) <= 1.0 + 1e-6)

    def test_returns_float32(self):
        """Output is always float32."""
        audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert peak_normalize(audio).dtype == np.float32


# ===========================================================================
# normalize (combined pipeline)
# ===========================================================================


class TestNormalize:
    def test_int16_stereo_normalized(self):
        """Full pipeline: int16 stereo → float32 mono in [-1, 1]."""
        stereo = np.array([[0, 16_384, -32_768], [0, 16_384, -32_768]], dtype=np.int16)
        result = normalize(stereo)
        assert result.ndim == 1
        assert result.dtype == np.float32
        assert np.all(np.abs(result) <= 1.0 + 1e-6)

    def test_no_peak_norm(self):
        """peak_norm=False preserves the float32 values after dtype conversion."""
        audio = np.array([0.3, -0.6], dtype=np.float32)
        result = normalize(audio, peak_norm=False)
        np.testing.assert_allclose(result, audio, atol=1e-7)

    def test_constants_exported(self):
        """Public constants have correct values."""
        assert CANONICAL_SAMPLE_RATE == 16_000
        assert CANONICAL_DTYPE == np.float32
