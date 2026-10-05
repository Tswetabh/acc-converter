"""
tests/integration/test_ecapa_integration.py
===========================================
Genuine integration tests for EcapaSpeakerEncoder using the real
``speechbrain/spkrec-ecapa-voxceleb`` checkpoint.

These tests:
  - Download / load the real ECAPA-TDNN model on first run.
  - Run real inference and verify the (192,) output shape and float32 dtype.
  - Verify deterministic / reproducible inference.
  - Verify distinct embeddings for different acoustic signals.
  - Benchmark inference latency.

Run separately:
    python -m pytest tests/integration/test_ecapa_integration.py -v -m integration

The normal unit test suite does NOT include these tests to avoid forced model downloads.
"""

from __future__ import annotations

import time
import numpy as np
import pytest
import torch

from accent_converter.models.backends.speaker.ecapa import EcapaSpeakerEncoder

MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
SAMPLE_RATE = 16_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _harmonic_signal(fundamental_hz: float, n_samples: int) -> np.ndarray:
    """Generate harmonic signal with multiple overtones (speaker proxy)."""
    t = np.linspace(0, n_samples / SAMPLE_RATE, n_samples, endpoint=False, dtype=np.float32)
    signal = (
        np.sin(2 * np.pi * fundamental_hz * t)
        + 0.5 * np.sin(2 * np.pi * (2 * fundamental_hz) * t)
        + 0.25 * np.sin(2 * np.pi * (3 * fundamental_hz) * t)
    ).astype(np.float32)
    # Normalize peak
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal /= max_val
    return signal * 0.8


def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


# ---------------------------------------------------------------------------
# Session-scoped fixture: load the model once
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def encoder():
    """Load ECAPA-TDNN model once for the integration test session."""
    enc = EcapaSpeakerEncoder()
    t0 = time.perf_counter()
    enc.load(MODEL_ID)
    load_time = time.perf_counter() - t0

    print(f"\n[ECAPA] Model loaded in {load_time:.2f}s on device={enc._device!r}")
    print(f"[ECAPA] Checkpoint: {enc._checkpoint_path!r}")
    return enc


# ===========================================================================
# Load verification
# ===========================================================================


@pytest.mark.integration
class TestModelLoading:
    """Verify model loads successfully."""

    def test_loaded_flag_is_true(self, encoder):
        assert encoder._loaded is True

    def test_classifier_is_not_none(self, encoder):
        assert encoder._classifier is not None

    def test_device_is_set(self, encoder):
        assert encoder._device in ("cpu", "cuda")

    def test_embedding_dim_is_192(self, encoder):
        assert encoder.embedding_dim == 192


# ===========================================================================
# Output shape & statistics
# ===========================================================================


@pytest.mark.integration
class TestOutputContract:
    """Verify embedding shape, dtype, and numerical properties."""

    def test_output_shape_and_dtype(self, encoder):
        audio = _harmonic_signal(120.0, 2 * SAMPLE_RATE)  # 2s male pitch proxy
        emb = encoder.encode(audio)

        assert isinstance(emb, np.ndarray), f"Expected np.ndarray, got {type(emb)}"
        assert emb.dtype == np.float32, f"Expected float32, got {emb.dtype}"
        assert emb.shape == (192,), f"Expected shape (192,), got {emb.shape}"

        l2_norm = float(np.linalg.norm(emb))
        print(f"\n[ECAPA] Embedding stats: min={emb.min():.4f}, max={emb.max():.4f}, L2-norm={l2_norm:.4f}")
        assert np.isfinite(emb).all()
        assert l2_norm > 0.0

    def test_determinism(self, encoder):
        audio = _harmonic_signal(220.0, SAMPLE_RATE)
        emb1 = encoder.encode(audio)
        emb2 = encoder.encode(audio)
        np.testing.assert_allclose(emb1, emb2, atol=1e-5)


# ===========================================================================
# Embedding discrimination (sanity checks)
# ===========================================================================


@pytest.mark.integration
class TestDiscrimination:
    """Verify different acoustic signatures yield distinct embeddings."""

    def test_distinct_signals_have_lower_cosine_similarity(self, encoder):
        # 100 Hz vs 400 Hz harmonic series proxy
        audio_low = _harmonic_signal(100.0, 2 * SAMPLE_RATE)
        audio_high = _harmonic_signal(400.0, 2 * SAMPLE_RATE)

        emb_low = encoder.encode(audio_low)
        emb_high = encoder.encode(audio_high)

        cos_sim = _cosine_similarity(emb_low, emb_high)
        print(f"\n[ECAPA] Cross-signal cosine similarity: {cos_sim:.4f}")
        assert cos_sim < 0.98, f"Expected distinct embeddings (< 0.98), got {cos_sim}"

    def test_same_signal_cross_utterance_similarity(self, encoder):
        # Same fundamental pitch, different duration (2s vs 3s)
        audio_2s = _harmonic_signal(150.0, 2 * SAMPLE_RATE)
        audio_3s = _harmonic_signal(150.0, 3 * SAMPLE_RATE)

        emb_2s = encoder.encode(audio_2s)
        emb_3s = encoder.encode(audio_3s)

        cos_sim = _cosine_similarity(emb_2s, emb_3s)
        print(f"\n[ECAPA] Same-signal (different length) cosine similarity: {cos_sim:.4f}")
        assert cos_sim > 0.5, f"Expected similar embeddings (> 0.5), got {cos_sim}"


# ===========================================================================
# Latency benchmark
# ===========================================================================


@pytest.mark.integration
class TestBenchmark:
    """Measure inference latency."""

    def test_cpu_inference_latency(self, encoder):
        audio = _harmonic_signal(180.0, SAMPLE_RATE)  # 1s

        # Warmup
        encoder.encode(audio)

        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            encoder.encode(audio)
            times.append(time.perf_counter() - t0)

        mean_ms = np.mean(times) * 1000
        std_ms = np.std(times) * 1000
        print(f"\n[ECAPA] 1.0s audio encode latency: {mean_ms:.2f} ms ± {std_ms:.2f} ms")
        assert mean_ms > 0
