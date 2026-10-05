"""
tests/unit/test_offline_pipeline.py
===================================
Unit tests for the full-utterance offline pipeline (Phase 6A).

All unit tests use mock models or stub synthesizers with temporary WAV files.
Zero network downloads and fast execution.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest
import soundfile as sf

from accent_converter.pipeline.offline import run_offline_pipeline
from accent_converter.models.interfaces import (
    ContentEncoder,
    SpeakerEncoder,
    AccentTranslator,
    SpeechSynthesizer,
)
from accent_converter.audio.preprocessing.interfaces import NoiseSuppressor
from accent_converter.audio.normalizer import CANONICAL_SAMPLE_RATE


# ---------------------------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------------------------


def _write_wav(path: Path, data: np.ndarray, sr: int = 16000) -> Path:
    """Helper to write a temporary wav file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sr)
    return path


@pytest.fixture
def temp_audio_dir(tmp_path: Path):
    """Create test source and enrollment audio files."""
    t = np.arange(16000, dtype=np.float32) / 16000.0
    source_wav = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    enrollment_wav = (0.4 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)

    src_file = _write_wav(tmp_path / "source.wav", source_wav, 16000)
    ref_file = _write_wav(tmp_path / "enrollment.wav", enrollment_wav, 16000)
    return tmp_path, src_file, ref_file


@pytest.fixture
def mock_pipeline_components():
    """Create mock components conforming to the abstract base classes."""
    mock_content = MagicMock(spec=ContentEncoder)
    mock_content.encode.return_value = np.zeros((50, 768), dtype=np.float32)

    mock_speaker = MagicMock(spec=SpeakerEncoder)
    mock_speaker.encode.return_value = np.zeros(192, dtype=np.float32)

    mock_translator = MagicMock(spec=AccentTranslator)
    mock_translator.translate.side_effect = lambda rep, target_accent: rep.copy()

    mock_synthesizer = MagicMock(spec=SpeechSynthesizer)
    mock_synthesizer.synthesize.return_value = np.zeros(50 * 320, dtype=np.float32)

    mock_ns = MagicMock(spec=NoiseSuppressor)
    mock_ns.process.side_effect = lambda audio: audio.copy()

    return {
        "content_encoder": mock_content,
        "speaker_encoder": mock_speaker,
        "accent_translator": mock_translator,
        "synthesizer": mock_synthesizer,
        "noise_suppressor": mock_ns,
    }


# ---------------------------------------------------------------------------
# Validation Tests
# ---------------------------------------------------------------------------


class TestOfflinePipelineValidation:
    def test_invalid_target_accent_raises_value_error(self, temp_audio_dir):
        _, src, ref = temp_audio_dir
        with pytest.raises(ValueError, match="target_accent"):
            run_offline_pipeline(src, "french", ref)

    def test_missing_source_file_raises_file_not_found(self, temp_audio_dir):
        tmp, _, ref = temp_audio_dir
        with pytest.raises(FileNotFoundError, match="Source audio"):
            run_offline_pipeline(tmp / "nonexistent.wav", "us", ref)

    def test_missing_enrollment_file_raises_file_not_found(self, temp_audio_dir):
        tmp, src, _ = temp_audio_dir
        with pytest.raises(FileNotFoundError, match="Enrollment audio"):
            run_offline_pipeline(src, "us", tmp / "nonexistent.wav")

    def test_missing_config_file_raises_file_not_found(self, temp_audio_dir):
        tmp, src, ref = temp_audio_dir
        with pytest.raises(FileNotFoundError, match="Config file"):
            run_offline_pipeline(src, "us", ref, config_path=tmp / "missing_config.yaml")

    def test_empty_source_audio_raises_value_error(self, tmp_path: Path):
        empty_src = _write_wav(tmp_path / "empty_src.wav", np.empty(0, dtype=np.float32))
        ref = _write_wav(tmp_path / "valid_ref.wav", np.zeros(16000, dtype=np.float32))
        with pytest.raises(ValueError, match="Source audio.*empty"):
            run_offline_pipeline(empty_src, "us", ref)

    def test_empty_enrollment_audio_raises_value_error(self, tmp_path: Path):
        src = _write_wav(tmp_path / "valid_src.wav", np.zeros(16000, dtype=np.float32))
        empty_ref = _write_wav(tmp_path / "empty_ref.wav", np.empty(0, dtype=np.float32))
        with pytest.raises(ValueError, match="Enrollment audio.*empty"):
            run_offline_pipeline(src, "us", empty_ref)


# ---------------------------------------------------------------------------
# Execution & Flow Tests
# ---------------------------------------------------------------------------


class TestOfflinePipelineExecution:
    def test_successful_run_with_injected_components(
        self, temp_audio_dir, mock_pipeline_components
    ):
        _, src, ref = temp_audio_dir
        waveform = run_offline_pipeline(
            audio_path=src,
            target_accent="us",
            enrollment_audio_path=ref,
            **mock_pipeline_components,
        )

        assert isinstance(waveform, np.ndarray)
        assert waveform.ndim == 1
        assert waveform.dtype == np.float32
        assert len(waveform) == 50 * 320

        # Verify calls
        mock_pipeline_components["speaker_encoder"].encode.assert_called_once()
        mock_pipeline_components["noise_suppressor"].process.assert_called_once()
        mock_pipeline_components["content_encoder"].encode.assert_called_once()
        mock_pipeline_components["accent_translator"].translate.assert_called_once()
        mock_pipeline_components["synthesizer"].synthesize.assert_called_once()

    def test_case_insensitive_accent(
        self, temp_audio_dir, mock_pipeline_components
    ):
        _, src, ref = temp_audio_dir
        run_offline_pipeline(
            audio_path=src,
            target_accent="  UK ",
            enrollment_audio_path=ref,
            **mock_pipeline_components,
        )
        _, kwargs = mock_pipeline_components["accent_translator"].translate.call_args
        assert kwargs["target_accent"] == "uk"

    def test_output_audio_file_written(
        self, temp_audio_dir, mock_pipeline_components
    ):
        tmp, src, ref = temp_audio_dir
        out_wav = tmp / "output_subdir" / "converted.wav"

        waveform = run_offline_pipeline(
            audio_path=src,
            target_accent="neutral",
            enrollment_audio_path=ref,
            output_path=out_wav,
            **mock_pipeline_components,
        )

        assert out_wav.is_file()
        loaded, sr = sf.read(str(out_wav), dtype="float32")
        assert sr == CANONICAL_SAMPLE_RATE
        np.testing.assert_array_equal(loaded, waveform)

    def test_resampling_applied_if_source_not_16k(
        self, tmp_path: Path, mock_pipeline_components
    ):
        """Input at 48 kHz is properly accepted and resampled."""
        t = np.arange(48000, dtype=np.float32) / 48000.0
        wav_48k = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        src_48k = _write_wav(tmp_path / "source_48k.wav", wav_48k, 48000)
        ref_16k = _write_wav(tmp_path / "ref_16k.wav", np.zeros(16000, dtype=np.float32), 16000)

        waveform = run_offline_pipeline(
            audio_path=src_48k,
            target_accent="us",
            enrollment_audio_path=ref_16k,
            **mock_pipeline_components,
        )
        assert len(waveform) > 0
