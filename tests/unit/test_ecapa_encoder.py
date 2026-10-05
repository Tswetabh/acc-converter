"""
tests/unit/test_ecapa_encoder.py
================================
Unit tests for EcapaSpeakerEncoder.

These tests do NOT download or load the real ECAPA-TDNN model.
All tests exercise:
  - Interface compliance & factory registration
  - Input validation (shape, dtype, empty, non-finite)
  - Unloaded-state guards
  - Metadata properties (embedding_dim = 192)
  - Device selection & validation
  - Mocked encode behavior & output contract

Real-model integration tests live in:
  tests/integration/test_ecapa_integration.py
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import torch

from accent_converter.models.backends.speaker.ecapa import EcapaSpeakerEncoder
from accent_converter.models.factory import build_speaker_encoder
from accent_converter.models.interfaces import SpeakerEncoder


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------

def _unloaded_encoder() -> EcapaSpeakerEncoder:
    """Return a fresh, unloaded EcapaSpeakerEncoder."""
    return EcapaSpeakerEncoder()


def _valid_audio(n_samples: int = 16_000) -> np.ndarray:
    """Return a valid 1-D float32 sine wave at 16 kHz."""
    t = np.linspace(0, n_samples / 16_000, n_samples, endpoint=False, dtype=np.float32)
    return np.sin(2 * np.pi * 440.0 * t).astype(np.float32)


@pytest.fixture
def mock_classifier():
    """Mock SpeechBrain EncoderClassifier."""
    classifier = MagicMock()
    # SpeechBrain ECAPA-TDNN encode_batch returns tensor of shape (batch, 1, 192)
    classifier.encode_batch.side_effect = lambda wavs: torch.ones(
        (wavs.shape[0], 1, 192), dtype=torch.float32
    ) * 0.5
    return classifier


@pytest.fixture
def patched_encoder(mock_classifier) -> EcapaSpeakerEncoder:
    """Return an EcapaSpeakerEncoder with a mocked SpeechBrain classifier."""
    enc = EcapaSpeakerEncoder()
    with patch(
        "speechbrain.inference.classifiers.EncoderClassifier.from_hparams",
        return_value=mock_classifier,
    ):
        enc.load("mock-checkpoint", device="cpu")
    return enc


# ===========================================================================
# Interface compliance & factory
# ===========================================================================


class TestInterfaceCompliance:
    """EcapaSpeakerEncoder must be a proper SpeakerEncoder subclass."""

    def test_is_speaker_encoder_subclass(self):
        assert issubclass(EcapaSpeakerEncoder, SpeakerEncoder)

    def test_is_speaker_encoder_instance(self):
        assert isinstance(_unloaded_encoder(), SpeakerEncoder)

    def test_backend_name(self):
        assert EcapaSpeakerEncoder.BACKEND_NAME == "ecapa"

    def test_embedding_dim_property(self):
        enc = _unloaded_encoder()
        assert enc.embedding_dim == 192

    def test_factory_builds_ecapa_encoder(self):
        enc = build_speaker_encoder({"backend": "ecapa", "checkpoint": None})
        assert isinstance(enc, EcapaSpeakerEncoder)
        assert enc.BACKEND_NAME == "ecapa"


# ===========================================================================
# Unloaded-state guards
# ===========================================================================


class TestUnloadedGuards:
    """encode() must fail safely when called before load()."""

    def test_encode_before_load_raises_runtime_error(self):
        enc = _unloaded_encoder()
        with pytest.raises(RuntimeError, match="called before load"):
            enc.encode(_valid_audio())


# ===========================================================================
# Input validation
# ===========================================================================


class TestInputValidation:
    """EcapaSpeakerEncoder must validate all audio inputs strictly."""

    def test_non_array_audio_raises_type_error(self, patched_encoder):
        with pytest.raises(TypeError, match="numpy.ndarray"):
            patched_encoder.encode([0.1, 0.2, 0.3])  # type: ignore

    def test_2d_audio_raises_value_error(self, patched_encoder):
        audio_2d = np.zeros((1, 1600), dtype=np.float32)
        with pytest.raises(ValueError, match="1-D"):
            patched_encoder.encode(audio_2d)

    def test_empty_array_raises_value_error(self, patched_encoder):
        with pytest.raises(ValueError, match="empty"):
            patched_encoder.encode(np.zeros(0, dtype=np.float32))

    def test_wrong_dtype_int16_raises_type_error(self, patched_encoder):
        audio_int16 = np.zeros(1600, dtype=np.int16)
        with pytest.raises(TypeError, match="float32"):
            patched_encoder.encode(audio_int16)

    def test_wrong_dtype_float64_raises_type_error(self, patched_encoder):
        audio_f64 = np.zeros(1600, dtype=np.float64)
        with pytest.raises(TypeError, match="float32"):
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


# ===========================================================================
# Inference & Output contract (mocked)
# ===========================================================================


class TestMockedInference:
    """Verify encode() output contract when backend returns embeddings."""

    def test_encode_returns_192_dim_float32(self, patched_encoder):
        audio = _valid_audio(8000)
        emb = patched_encoder.encode(audio)
        assert isinstance(emb, np.ndarray)
        assert emb.dtype == np.float32
        assert emb.shape == (192,)

    def test_encode_unexpected_dimension_raises_assertion_error(self, mock_classifier):
        enc = EcapaSpeakerEncoder()
        # Return wrong dimension (e.g. 256 instead of 192)
        mock_classifier.encode_batch.side_effect = lambda wavs: torch.zeros(
            (wavs.shape[0], 1, 256), dtype=torch.float32
        )
        with patch(
            "speechbrain.inference.classifiers.EncoderClassifier.from_hparams",
            return_value=mock_classifier,
        ):
            enc.load("mock-ckpt", device="cpu")

        with pytest.raises(AssertionError, match="Expected speaker embedding shape"):
            enc.encode(_valid_audio())


# ===========================================================================
# reset() safety
# ===========================================================================


class TestResetSafety:
    """reset() must be a no-op and never raise."""

    def test_reset_before_load_is_safe(self):
        enc = _unloaded_encoder()
        enc.reset()

    def test_reset_after_load_is_safe(self, patched_encoder):
        patched_encoder.reset()
        assert patched_encoder._loaded is True


# ===========================================================================
# Device selection & Load error handling
# ===========================================================================


class TestDeviceSelectionAndErrors:
    """Verify device validation and load error handling."""

    def test_invalid_device_raises_value_error(self):
        enc = _unloaded_encoder()
        with pytest.raises(ValueError, match="Invalid device"):
            enc.load("mock-ckpt", device="invalid_device_xyz")

    def test_load_failure_raises_runtime_error(self):
        enc = _unloaded_encoder()
        with patch(
            "speechbrain.inference.classifiers.EncoderClassifier.from_hparams",
            side_effect=RuntimeError("Download failed"),
        ):
            with pytest.raises(RuntimeError, match="Failed to load ECAPA-TDNN"):
                enc.load("mock-ckpt", device="cpu")
