"""
tests/unit/test_silero_vad.py
=============================
Unit tests for the neural SileroVAD Voice Activity Detector backend.

All unit tests use mocked or local components — zero network downloads.
"""

from __future__ import annotations

from unittest.mock import MagicMock
import numpy as np
import pytest
import torch

from accent_converter.audio.preprocessing.interfaces import (
    VoiceActivityDetector,
    VADResult,
)
from accent_converter.audio.preprocessing.factory import build_vad
from accent_converter.audio.preprocessing.vad.silero import (
    SileroVAD,
    DEFAULT_SILERO_THRESHOLD,
    DEFAULT_HANGOVER_FRAMES,
    WINDOW_SIZE_16K,
    WINDOW_SIZE_8K,
)
from accent_converter.audio.chunker import CHUNK_SAMPLES


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------


def _sine_chunk(n: int = CHUNK_SAMPLES, amp: float = 0.5) -> np.ndarray:
    """Return a 1-D float32 sine wave at 16 kHz."""
    t = np.arange(n, dtype=np.float32) / 16000.0
    return (amp * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)


def _silence_chunk(n: int = CHUNK_SAMPLES) -> np.ndarray:
    """Return an all-zero float32 chunk."""
    return np.zeros(n, dtype=np.float32)


@pytest.fixture
def mock_silero_model():
    """Create a mock Silero VAD neural model with reset_states."""
    model = MagicMock()
    model.reset_states = MagicMock()
    # Default: low speech probability (0.01)
    model.side_effect = lambda tensor_win, sr: torch.tensor(0.01)
    return model


@pytest.fixture
def mocked_vad(mock_silero_model) -> SileroVAD:
    """Return a SileroVAD instance using the mocked model on CPU."""
    return SileroVAD(
        threshold=0.5,
        hangover_frames=DEFAULT_HANGOVER_FRAMES,
        sampling_rate=16000,
        device="cpu",
        model=mock_silero_model,
    )


# ---------------------------------------------------------------------------
# Construction & Validation
# ---------------------------------------------------------------------------


class TestSileroVADConstruction:
    def test_default_parameters(self, mock_silero_model):
        vad = SileroVAD(device="cpu", model=mock_silero_model)
        assert vad.threshold == DEFAULT_SILERO_THRESHOLD
        assert vad.hangover_frames == DEFAULT_HANGOVER_FRAMES
        assert vad.sampling_rate == 16000
        assert vad.window_size == WINDOW_SIZE_16K
        assert vad.device.type == "cpu"
        assert vad.BACKEND_NAME == "silero"
        assert isinstance(vad, VoiceActivityDetector)

    def test_custom_parameters(self, mock_silero_model):
        vad = SileroVAD(
            threshold=0.75,
            hangover_frames=4,
            sampling_rate=8000,
            device="cpu",
            model=mock_silero_model,
        )
        assert vad.threshold == 0.75
        assert vad.hangover_frames == 4
        assert vad.sampling_rate == 8000
        assert vad.window_size == WINDOW_SIZE_8K

    def test_invalid_threshold_raises(self, mock_silero_model):
        with pytest.raises(ValueError, match="threshold"):
            SileroVAD(threshold=-0.1, model=mock_silero_model)
        with pytest.raises(ValueError, match="threshold"):
            SileroVAD(threshold=1.1, model=mock_silero_model)

    def test_negative_hangover_raises(self, mock_silero_model):
        with pytest.raises(ValueError, match="hangover_frames"):
            SileroVAD(hangover_frames=-1, model=mock_silero_model)

    def test_unsupported_sample_rate_raises(self, mock_silero_model):
        with pytest.raises(ValueError, match="sampling_rate"):
            SileroVAD(sampling_rate=44100, model=mock_silero_model)


# ---------------------------------------------------------------------------
# Speech Detection & Timing Preservation
# ---------------------------------------------------------------------------


class TestSileroVADInference:
    def test_speech_detected_when_prob_above_threshold(self, mocked_vad, mock_silero_model):
        mock_silero_model.side_effect = lambda t, sr: torch.tensor(0.85)
        result = mocked_vad.process(_sine_chunk())
        assert isinstance(result, VADResult)
        assert result.is_speech is True

    def test_silence_detected_when_prob_below_threshold(self, mocked_vad, mock_silero_model):
        mock_silero_model.side_effect = lambda t, sr: torch.tensor(0.05)
        mocked_vad.hangover_frames = 0
        result = mocked_vad.process(_silence_chunk())
        assert result.is_speech is False

    def test_audio_unmodified_identity(self, mocked_vad):
        chunk = _sine_chunk()
        result = mocked_vad.process(chunk)
        assert result.audio is chunk
        np.testing.assert_array_equal(result.audio, chunk)

    def test_energy_db_is_computed(self, mocked_vad):
        result = mocked_vad.process(_sine_chunk(amp=0.5))
        assert isinstance(result.energy_db, float)
        assert -10.0 <= result.energy_db <= 0.0


# ---------------------------------------------------------------------------
# Hangover Logic
# ---------------------------------------------------------------------------


class TestSileroVADHangover:
    def test_hangover_holds_speech_label(self, mocked_vad, mock_silero_model):
        hangover = 3
        mocked_vad.hangover_frames = hangover

        # Frame 1: speech
        mock_silero_model.side_effect = lambda t, sr: torch.tensor(0.9)
        r = mocked_vad.process(_sine_chunk())
        assert r.is_speech is True

        # Next 3 frames: non-speech prob, but hangover holds label
        mock_silero_model.side_effect = lambda t, sr: torch.tensor(0.01)
        for i in range(hangover):
            r = mocked_vad.process(_silence_chunk())
            assert r.is_speech is True, f"Frame {i+1} should hold speech"

        # Frame 5: hangover expired
        r = mocked_vad.process(_silence_chunk())
        assert r.is_speech is False

    def test_new_speech_reloads_hangover(self, mocked_vad, mock_silero_model):
        hangover = 3
        mocked_vad.hangover_frames = hangover

        # Speech
        mock_silero_model.side_effect = lambda t, sr: torch.tensor(0.9)
        mocked_vad.process(_sine_chunk())

        # 2 frames silence
        mock_silero_model.side_effect = lambda t, sr: torch.tensor(0.01)
        mocked_vad.process(_silence_chunk())
        mocked_vad.process(_silence_chunk())

        # Speech again -> counter reloaded
        mock_silero_model.side_effect = lambda t, sr: torch.tensor(0.9)
        mocked_vad.process(_sine_chunk())

        # Silence for full hangover duration
        mock_silero_model.side_effect = lambda t, sr: torch.tensor(0.01)
        for _ in range(hangover):
            assert mocked_vad.process(_silence_chunk()).is_speech is True

        assert mocked_vad.process(_silence_chunk()).is_speech is False

    def test_zero_hangover_flips_immediately(self, mocked_vad, mock_silero_model):
        mocked_vad.hangover_frames = 0

        mock_silero_model.side_effect = lambda t, sr: torch.tensor(0.9)
        assert mocked_vad.process(_sine_chunk()).is_speech is True

        mock_silero_model.side_effect = lambda t, sr: torch.tensor(0.01)
        assert mocked_vad.process(_silence_chunk()).is_speech is False


# ---------------------------------------------------------------------------
# Streaming Window Buffer
# ---------------------------------------------------------------------------


class TestSileroVADStreamingBuffer:
    def test_windowing_evaluations_across_chunks(self, mocked_vad, mock_silero_model):
        """1280 samples at 16 kHz: chunk 0 runs 2 windows (leftover 256), chunk 1 runs 3 windows (leftover 0)."""
        call_count = 0

        def _count_calls(t, sr):
            nonlocal call_count
            call_count += 1
            assert t.shape[-1] == 512
            return torch.tensor(0.1)

        mock_silero_model.side_effect = _count_calls

        # Chunk 0: 1280 samples -> 2 windows of 512 (1024 total), 256 in buffer
        mocked_vad.process(_sine_chunk(n=1280))
        assert call_count == 2
        assert len(mocked_vad._buffer) == 256

        # Chunk 1: 1280 samples -> 256 + 1280 = 1536 samples -> 3 windows of 512, 0 in buffer
        mocked_vad.process(_sine_chunk(n=1280))
        assert call_count == 5
        assert len(mocked_vad._buffer) == 0


# ---------------------------------------------------------------------------
# Reset Behavior
# ---------------------------------------------------------------------------


class TestSileroVADReset:
    def test_reset_clears_buffer_hangover_and_model_states(self, mocked_vad, mock_silero_model):
        mock_silero_model.side_effect = lambda t, sr: torch.tensor(0.9)
        mocked_vad.process(_sine_chunk(n=1280))
        assert len(mocked_vad._buffer) > 0
        assert mocked_vad._hangover_counter > 0

        mocked_vad.reset()

        assert len(mocked_vad._buffer) == 0
        assert mocked_vad._hangover_counter == 0
        mock_silero_model.reset_states.assert_called_once()

    def test_reset_before_any_call_is_safe(self, mocked_vad):
        mocked_vad.reset()


# ---------------------------------------------------------------------------
# Edge Cases & Input Validation
# ---------------------------------------------------------------------------


class TestSileroVADEdgeCases:
    def test_non_array_input_raises_type_error(self, mocked_vad):
        with pytest.raises(TypeError, match="numpy ndarray"):
            mocked_vad.process([0.1, 0.2, 0.3])  # type: ignore[arg-type]

    def test_2d_input_raises_value_error(self, mocked_vad):
        with pytest.raises(ValueError, match="1-D"):
            mocked_vad.process(np.zeros((2, 1280), dtype=np.float32))

    def test_non_floating_dtype_raises_type_error(self, mocked_vad):
        with pytest.raises(TypeError, match="floating-point"):
            mocked_vad.process(np.zeros(1280, dtype=np.int32))

    def test_nan_or_inf_raises_value_error(self, mocked_vad):
        bad_nan = np.zeros(1280, dtype=np.float32)
        bad_nan[10] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            mocked_vad.process(bad_nan)

        bad_inf = np.zeros(1280, dtype=np.float32)
        bad_inf[10] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            mocked_vad.process(bad_inf)

    def test_empty_chunk_returns_silence(self, mocked_vad):
        result = mocked_vad.process(np.zeros(0, dtype=np.float32))
        assert result.is_speech is False
        assert result.energy_db <= -90.0
        assert len(result.audio) == 0


# ---------------------------------------------------------------------------
# Factory Integration
# ---------------------------------------------------------------------------


class TestSileroVADFactory:
    def test_build_vad_silero(self, mock_silero_model, monkeypatch):
        import silero_vad
        monkeypatch.setattr(silero_vad, "load_silero_vad", lambda *a, **kw: mock_silero_model)

        vad = build_vad({"backend": "silero"})
        assert isinstance(vad, VoiceActivityDetector)
        assert isinstance(vad, SileroVAD)
        assert vad.threshold == DEFAULT_SILERO_THRESHOLD

    def test_build_vad_silero_with_kwargs(self, mock_silero_model, monkeypatch):
        import silero_vad
        monkeypatch.setattr(silero_vad, "load_silero_vad", lambda *a, **kw: mock_silero_model)

        vad = build_vad({
            "backend": "silero",
            "threshold": 0.65,
            "hangover_frames": 4,
            "device": "cpu",
        })
        assert isinstance(vad, SileroVAD)
        assert vad.threshold == 0.65
        assert vad.hangover_frames == 4
        assert vad.device.type == "cpu"
