"""
tests/integration/test_hubert_integration.py
=============================================
Genuine integration tests for HubertContentEncoder using the real
``facebook/hubert-base-ls960`` checkpoint.

These tests:
  - Download / load the real HuBERT checkpoint on first run (~360 MB).
  - Run real inference and verify the (T, 768) output shape.
  - Verify deterministic / reproducible inference.
  - Test multiple audio durations and REPORT actual frame counts.

DO NOT assume fixed frame counts (e.g. "1 second → 49 frames").
The actual T value is measured from the model and reported.

Run separately:
    python -m pytest tests/integration/test_hubert_integration.py -v

The normal pytest suite (pytest.ini testpaths = tests/unit) does NOT
include these tests to avoid forced model downloads.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

MODEL_ID = "facebook/hubert-base-ls960"
SAMPLE_RATE = 16_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sine(freq_hz: float, n_samples: int) -> np.ndarray:
    """Deterministic float32 sine wave at 16 kHz."""
    t = np.linspace(0, n_samples / SAMPLE_RATE, n_samples, endpoint=False, dtype=np.float32)
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float32)


def _silence(n_samples: int) -> np.ndarray:
    return np.zeros(n_samples, dtype=np.float32)


# ---------------------------------------------------------------------------
# Session-scoped fixture: load the model exactly once per test session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def encoder():
    """Load facebook/hubert-base-ls960 once for the entire test session."""
    from accent_converter.models.backends.content.hubert import HubertContentEncoder

    enc = HubertContentEncoder()
    t0 = time.perf_counter()
    enc.load(MODEL_ID)
    load_time = time.perf_counter() - t0

    print(f"\n[HuBERT] Model loaded in {load_time:.2f}s on device={enc._device!r}")
    print(f"[HuBERT] Checkpoint: {enc._checkpoint_path!r}")

    return enc


# ===========================================================================
# Load verification
# ===========================================================================


@pytest.mark.integration
class TestModelLoading:
    """Verify the model loaded successfully."""

    def test_loaded_flag_is_true(self, encoder):
        assert encoder._loaded is True

    def test_processor_is_not_none(self, encoder):
        assert encoder._processor is not None

    def test_model_is_not_none(self, encoder):
        assert encoder._model is not None

    def test_device_is_set(self, encoder):
        assert encoder._device in ("cpu", "cuda")

    def test_metadata_properties_post_load(self, encoder):
        assert encoder.frame_rate_hz == 50
        assert encoder.output_dim == 768
        assert encoder.lookahead_ms == -1


# ===========================================================================
# Output shape verification
# ===========================================================================


@pytest.mark.integration
class TestOutputShape:
    """Verify encode() returns (T_frames, 768) float32 numpy arrays."""

    def test_output_is_numpy_array(self, encoder):
        audio = _sine(440.0, SAMPLE_RATE)  # 1 second
        features = encoder.encode(audio)
        assert isinstance(features, np.ndarray), (
            f"Expected np.ndarray, got {type(features).__name__}"
        )

    def test_output_is_2d(self, encoder):
        audio = _sine(440.0, SAMPLE_RATE)
        features = encoder.encode(audio)
        assert features.ndim == 2, f"Expected 2-D output, got shape {features.shape}"

    def test_output_dim_is_768(self, encoder):
        audio = _sine(440.0, SAMPLE_RATE)
        features = encoder.encode(audio)
        assert features.shape[1] == 768, (
            f"Expected output_dim=768, got {features.shape[1]}"
        )

    def test_output_dtype_is_float32(self, encoder):
        audio = _sine(440.0, SAMPLE_RATE)
        features = encoder.encode(audio)
        assert features.dtype == np.float32, (
            f"Expected float32 output, got {features.dtype}"
        )

    def test_output_has_nonzero_frames(self, encoder):
        audio = _sine(440.0, SAMPLE_RATE)
        features = encoder.encode(audio)
        assert features.shape[0] > 0, "Expected at least one output frame"


# ===========================================================================
# Frame count measurement across durations
# ===========================================================================


@pytest.mark.integration
class TestFrameCounts:
    """Measure and report actual frame counts — do NOT assert fixed values."""

    DURATIONS_S = [0.5, 1.0, 2.0, 3.0, 5.0]

    def test_frame_counts_reported(self, encoder):
        """Encode multiple durations and print the T→duration relationship.

        No hard assertion on the frame count; only that:
        - T > 0 for each duration.
        - T increases monotonically with duration.
        - shape[-1] == 768 for all.
        """
        results = []
        for duration_s in self.DURATIONS_S:
            n = int(duration_s * SAMPLE_RATE)
            audio = _sine(440.0, n)
            features = encoder.encode(audio)
            t_frames = features.shape[0]
            approx_rate = t_frames / duration_s
            results.append((duration_s, t_frames, approx_rate))
            print(
                f"  [frame count] {duration_s:.1f}s audio "
                f"→ {t_frames} frames "
                f"(≈{approx_rate:.1f} frames/s)"
            )

        # Validate monotonicity
        frame_counts = [r[1] for r in results]
        for i in range(1, len(frame_counts)):
            assert frame_counts[i] > frame_counts[i - 1], (
                f"Frame count did not increase between "
                f"{self.DURATIONS_S[i-1]}s and {self.DURATIONS_S[i]}s"
            )

        # Validate dimension for all
        for duration_s in self.DURATIONS_S:
            n = int(duration_s * SAMPLE_RATE)
            audio = _sine(440.0, n)
            assert encoder.encode(audio).shape[1] == 768

    def test_silence_produces_valid_frames(self, encoder):
        """Silence audio must still produce a valid (T, 768) output."""
        audio = _silence(SAMPLE_RATE)  # 1 second of silence
        features = encoder.encode(audio)
        t_frames = features.shape[0]
        print(f"\n  [silence] 1.0s silence → {t_frames} frames, shape={features.shape}")
        assert features.ndim == 2
        assert features.shape[1] == 768
        assert t_frames > 0


# ===========================================================================
# Determinism / reproducibility
# ===========================================================================


@pytest.mark.integration
class TestDeterminism:
    """Identical inputs must produce identical outputs (model is eval mode)."""

    def test_same_input_same_output(self, encoder):
        audio = _sine(261.63, int(1.0 * SAMPLE_RATE))  # Middle C, 1s

        features_a = encoder.encode(audio)
        features_b = encoder.encode(audio)

        np.testing.assert_array_equal(
            features_a, features_b,
            err_msg=(
                "HuBERT encode() must be deterministic: identical inputs "
                "must produce identical outputs."
            ),
        )

    def test_different_inputs_different_outputs(self, encoder):
        """Sanity check: different audio should not produce identical features."""
        audio_a = _sine(440.0, SAMPLE_RATE)    # A4
        audio_b = _sine(880.0, SAMPLE_RATE)    # A5

        features_a = encoder.encode(audio_a)
        features_b = encoder.encode(audio_b)

        # Features should not be elementwise identical for different signals
        assert not np.array_equal(features_a, features_b), (
            "Different audio inputs produced identical features — "
            "this suggests the encoder is ignoring input."
        )

    def test_output_is_finite(self, encoder):
        """All output values must be finite (no nan or inf)."""
        audio = _sine(440.0, SAMPLE_RATE)
        features = encoder.encode(audio)
        assert np.all(np.isfinite(features)), (
            "HuBERT encode() produced non-finite values (nan or inf)."
        )
