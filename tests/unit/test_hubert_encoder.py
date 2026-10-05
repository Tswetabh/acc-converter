"""
tests/unit/test_hubert_encoder.py
===================================
Unit tests for HubertContentEncoder.

These tests do NOT download or load the real HuBERT model.
All tests exercise:
  - Input validation (shape, dtype, empty, non-finite)
  - Unloaded-state guards
  - Metadata properties
  - Device selection logic

Network-dependent / real-model tests live in:
  tests/integration/test_hubert_integration.py
"""

from __future__ import annotations

import pytest
import numpy as np

from accent_converter.models.backends.content.hubert import HubertContentEncoder
from accent_converter.models.interfaces import ContentEncoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unloaded_encoder() -> HubertContentEncoder:
    """Return a fresh, unloaded HubertContentEncoder."""
    return HubertContentEncoder()


def _valid_audio(n_samples: int = 16_000) -> np.ndarray:
    """Return a valid 1-D float32 sine wave at 16 kHz."""
    t = np.linspace(0, n_samples / 16_000, n_samples, endpoint=False, dtype=np.float32)
    return np.sin(2 * np.pi * 440.0 * t).astype(np.float32)


# ===========================================================================
# Interface compliance
# ===========================================================================


class TestInterfaceCompliance:
    """HubertContentEncoder must be a proper ContentEncoder subclass."""

    def test_is_content_encoder_subclass(self):
        assert issubclass(HubertContentEncoder, ContentEncoder)

    def test_is_content_encoder_instance(self):
        assert isinstance(_unloaded_encoder(), ContentEncoder)

    def test_backend_name(self):
        assert HubertContentEncoder.BACKEND_NAME == "hubert"


# ===========================================================================
# Metadata properties
# ===========================================================================


class TestMetadataProperties:
    """Verify HuBERT Base metadata constants — no model load required."""

    def test_frame_rate_hz_is_50(self):
        enc = _unloaded_encoder()
        assert enc.frame_rate_hz == 50

    def test_output_dim_is_768(self):
        enc = _unloaded_encoder()
        assert enc.output_dim == 768

    def test_lookahead_ms_is_minus_one(self):
        """Non-causal sentinel must be -1."""
        enc = _unloaded_encoder()
        assert enc.lookahead_ms == -1

    def test_all_metadata_are_ints(self):
        enc = _unloaded_encoder()
        assert isinstance(enc.frame_rate_hz, int)
        assert isinstance(enc.output_dim, int)
        assert isinstance(enc.lookahead_ms, int)

    def test_is_not_streaming_safe(self):
        """lookahead_ms >= 0 is the streaming-safe check; HuBERT must fail it."""
        enc = _unloaded_encoder()
        assert enc.lookahead_ms < 0, (
            "HuBERT must not pass the streaming-safe check (lookahead_ms >= 0)"
        )


# ===========================================================================
# Unloaded-state guard
# ===========================================================================


class TestUnloadedStateGuard:
    """encode() must raise RuntimeError when called before load()."""

    def test_encode_before_load_raises_runtime_error(self):
        enc = _unloaded_encoder()
        audio = _valid_audio()
        with pytest.raises(RuntimeError, match="called before load"):
            enc.encode(audio)

    def test_loaded_flag_starts_false(self):
        enc = _unloaded_encoder()
        assert enc._loaded is False

    def test_model_and_processor_start_as_none(self):
        enc = _unloaded_encoder()
        assert enc._model is None
        assert enc._processor is None
        assert enc._device is None


# ===========================================================================
# Input validation
# ===========================================================================


class TestInputValidation:
    """encode() must validate audio input and raise descriptive errors."""

    # These tests patch _loaded=True and inject a dummy processor/model so that
    # input validation is exercised without a real model.

    @pytest.fixture
    def patched_encoder(self):
        """HubertContentEncoder with _loaded=True but no real model.

        Only use for tests that want to reach input-validation code without
        triggering actual inference.
        """
        enc = HubertContentEncoder()
        enc._loaded = True
        # _model and _processor remain None; the test must not reach the
        # inference step.  Any test reaching inference with this fixture will
        # get an AttributeError — which is intentional.
        return enc

    def test_non_array_raises_value_error(self, patched_encoder):
        with pytest.raises(ValueError, match="numpy.ndarray"):
            patched_encoder.encode([0.1, 0.2, 0.3])  # list, not ndarray

    def test_2d_array_raises_value_error(self, patched_encoder):
        audio_2d = np.zeros((2, 16_000), dtype=np.float32)
        with pytest.raises(ValueError, match="1-D"):
            patched_encoder.encode(audio_2d)

    def test_empty_array_raises_value_error(self, patched_encoder):
        with pytest.raises(ValueError, match="empty"):
            patched_encoder.encode(np.zeros(0, dtype=np.float32))

    def test_wrong_dtype_int16_raises_value_error(self, patched_encoder):
        audio_int16 = np.zeros(1600, dtype=np.int16)
        with pytest.raises(ValueError, match="float32"):
            patched_encoder.encode(audio_int16)

    def test_wrong_dtype_float64_raises_value_error(self, patched_encoder):
        audio_f64 = np.zeros(1600, dtype=np.float64)
        with pytest.raises(ValueError, match="float32"):
            patched_encoder.encode(audio_f64)

    def test_nan_audio_raises_value_error(self, patched_encoder):
        audio = np.full(1600, np.nan, dtype=np.float32)
        with pytest.raises(ValueError, match="non-finite"):
            patched_encoder.encode(audio)

    def test_inf_audio_raises_value_error(self, patched_encoder):
        audio = np.full(1600, np.inf, dtype=np.float32)
        with pytest.raises(ValueError, match="non-finite"):
            patched_encoder.encode(audio)

    def test_negative_inf_raises_value_error(self, patched_encoder):
        audio = np.full(1600, -np.inf, dtype=np.float32)
        with pytest.raises(ValueError, match="non-finite"):
            patched_encoder.encode(audio)

    def test_mixed_nan_raises_value_error(self, patched_encoder):
        audio = _valid_audio()
        audio[100] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            patched_encoder.encode(audio)


# ===========================================================================
# reset() safety
# ===========================================================================


class TestResetSafety:
    """reset() must be a no-op and never raise at any point."""

    def test_reset_before_load_is_safe(self):
        _unloaded_encoder().reset()

    def test_double_reset_is_safe(self):
        enc = _unloaded_encoder()
        enc.reset()
        enc.reset()

    def test_reset_does_not_clear_loaded_state(self):
        """reset() on a loaded encoder must not clear _loaded.

        (HuBERT has no streaming state to clear — but reset() must be safe
        on a loaded encoder and must not accidentally destroy it.)
        """
        enc = _unloaded_encoder()
        enc._loaded = True  # simulate post-load state
        enc.reset()
        assert enc._loaded is True


# ===========================================================================
# Device selection
# ===========================================================================


class TestDeviceSelection:
    """Verify device-selection logic without loading a real model.

    These tests check that HubertContentEncoder uses explicit device override
    correctly and rejects invalid device strings.
    """

    def test_invalid_device_string_raises_on_load(self):
        """Passing a garbage device string must fail at load() time."""
        enc = _unloaded_encoder()
        with pytest.raises((ValueError, RuntimeError, OSError)):
            enc.load("non-existent-local-path-intentionally", device="invalid_device_xyz")
