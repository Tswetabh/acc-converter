"""
tests/integration/test_offline_pipeline.py
=========================================
End-to-end integration test for the full offline pipeline (Phase 6A).

This test wires all real model backends:
- HubertContentEncoder (facebook/hubert-base-ls960)
- EcapaSpeakerEncoder (speechbrain/spkrec-ecapa-voxceleb)
- NaiveAccentTranslator (passthrough)
- VocoderSynthesizer (Phase 6A duration-matching stub)
- NoiseReduceNoiseSuppressor

Marked with @pytest.mark.integration (opt-in; skipped during default unit tests).
"""

from __future__ import annotations

from pathlib import Path
import time
import numpy as np
import pytest
import soundfile as sf

from accent_converter.pipeline.offline import run_offline_pipeline
from accent_converter.audio.normalizer import CANONICAL_SAMPLE_RATE


@pytest.fixture
def sample_wav_files(tmp_path: Path):
    """Generate deterministic 1.5s source audio and 2.0s enrollment audio."""
    sr = 16000
    t_src = np.arange(int(1.5 * sr), dtype=np.float32) / sr
    t_ref = np.arange(int(2.0 * sr), dtype=np.float32) / sr

    # Source: 440 Hz tone + 880 Hz harmonic
    source_audio = (0.4 * np.sin(2 * np.pi * 440.0 * t_src) + 0.2 * np.sin(2 * np.pi * 880.0 * t_src)).astype(np.float32)
    # Enrollment: 220 Hz tone
    enrollment_audio = (0.5 * np.sin(2 * np.pi * 220.0 * t_ref)).astype(np.float32)

    src_path = tmp_path / "test_source.wav"
    ref_path = tmp_path / "test_enrollment.wav"
    out_path = tmp_path / "test_converted.wav"

    sf.write(str(src_path), source_audio, sr)
    sf.write(str(ref_path), enrollment_audio, sr)

    return src_path, ref_path, out_path, len(source_audio)


@pytest.mark.integration
class TestOfflinePipelineEndToEnd:
    def test_run_offline_pipeline_real_models(self, sample_wav_files):
        src_path, ref_path, out_path, expected_source_samples = sample_wav_files

        t0 = time.perf_counter()
        waveform = run_offline_pipeline(
            audio_path=src_path,
            target_accent="us",
            enrollment_audio_path=ref_path,
            output_path=out_path,
        )
        elapsed = time.perf_counter() - t0

        # Assert output type and dimensions
        assert isinstance(waveform, np.ndarray)
        assert waveform.ndim == 1
        assert waveform.dtype == np.float32

        # Check duration matches source within 10%
        # (1.5s = 24,000 samples -> HuBERT produces 73 frames -> 73 * 320 = 23,360 samples, ~2.6% difference)
        duration_ratio = len(waveform) / expected_source_samples
        assert 0.90 <= duration_ratio <= 1.10, (
            f"Expected duration ratio close to 1.0, got {duration_ratio:.3f}"
        )

        # Assert output file written and readable
        assert out_path.is_file()
        file_audio, file_sr = sf.read(str(out_path), dtype="float32")
        assert file_sr == CANONICAL_SAMPLE_RATE
        np.testing.assert_array_equal(file_audio, waveform)

        print(f"\n[Phase 6A Smoke Test] Full pipeline elapsed: {elapsed:.2f} s")
        print(f"[Phase 6A Smoke Test] Converted samples: {len(waveform)} ({len(waveform)/16000:.2f} s)")
