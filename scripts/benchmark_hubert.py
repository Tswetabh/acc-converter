"""
scripts/benchmark_hubert.py
============================
Benchmark HubertContentEncoder — model load time, inference latency, RTF,
output shapes, and (when available) CUDA VRAM usage.

Usage
-----
    python scripts/benchmark_hubert.py

What is measured
----------------
  - Model load time (seconds)
  - Inference latency on CPU (ms) for 1 s, 3 s, 5 s input
  - Inference latency on CUDA (ms) — only when torch.cuda.is_available()
  - Real-Time Factor (RTF) = processing_time / audio_duration
  - Actual output shapes (T_frames, 768)
  - Actual frame counts and derived frame rates
  - Peak CUDA memory allocated (MB) — when CUDA is available

CUDA benchmark only runs if torch.cuda.is_available().
CPU benchmark always runs.

All values are measured; nothing is estimated or invented.
"""

from __future__ import annotations

import sys
import os
import time

# Ensure src/ is importable
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_root, "src"))

import numpy as np

MODEL_ID = "facebook/hubert-base-ls960"
SAMPLE_RATE = 16_000
DURATIONS_S = [1.0, 3.0, 5.0]
N_REPEATS = 3   # number of repetitions for latency averaging


def _sine(freq_hz: float, duration_s: float) -> np.ndarray:
    n = int(duration_s * SAMPLE_RATE)
    t = np.linspace(0, duration_s, n, endpoint=False, dtype=np.float32)
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float32)


def _section(title: str) -> None:
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print(f"{'-' * 60}")


def benchmark_device(device: str) -> None:
    """Run inference benchmarks on the given device."""
    import torch

    # Reload encoder on the target device
    from accent_converter.models.backends.content.hubert import HubertContentEncoder
    enc = HubertContentEncoder()

    _section(f"Loading model on device={device!r}")
    t_load_start = time.perf_counter()
    enc.load(MODEL_ID, device=device)
    t_load = time.perf_counter() - t_load_start
    print(f"  Load time        : {t_load:.3f} s")
    print(f"  Device           : {enc._device}")
    print(f"  frame_rate_hz    : {enc.frame_rate_hz}")
    print(f"  output_dim       : {enc.output_dim}")
    print(f"  lookahead_ms     : {enc.lookahead_ms}  (offline-only, non-causal)")

    _section(f"Inference latency on {device.upper()}")
    header = (
        f"  {'Duration':>8s}  {'T_frames':>8s}  {'Frames/s':>10s}  "
        f"{'Latency ms':>12s}  {'RTF':>8s}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for duration_s in DURATIONS_S:
        audio = _sine(440.0, duration_s)

        # Warm-up pass (not timed)
        enc.encode(audio)

        if device == "cuda":
            torch.cuda.synchronize()

        # Timed passes
        latencies = []
        for _ in range(N_REPEATS):
            t0 = time.perf_counter()
            features = enc.encode(audio)
            if device == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1_000)

        mean_latency_ms = sum(latencies) / len(latencies)
        rtf = (mean_latency_ms / 1_000) / duration_s
        t_frames = features.shape[0]
        frames_per_s = t_frames / duration_s

        print(
            f"  {duration_s:>7.1f}s  {t_frames:>8d}  "
            f"{frames_per_s:>10.2f}  {mean_latency_ms:>12.2f}  {rtf:>8.4f}"
        )

    if device == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"\n  Peak CUDA memory allocated : {peak_mb:.1f} MB")
        torch.cuda.reset_peak_memory_stats()


def main() -> None:
    import torch

    print("=" * 60)
    print("  HuBERT Offline Content Encoder - Benchmark")
    print("=" * 60)
    print(f"  Model             : {MODEL_ID}")
    print(f"  Sample rate       : {SAMPLE_RATE} Hz")
    print(f"  Audio durations   : {DURATIONS_S} s")
    print(f"  Timing repeats    : {N_REPEATS}")
    print(f"  torch version     : {torch.__version__}")
    print(f"  CUDA available    : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU               : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM total        : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Always benchmark CPU
    benchmark_device("cpu")

    # Benchmark CUDA only if available
    if torch.cuda.is_available():
        benchmark_device("cuda")
    else:
        print("\n  [CUDA benchmark skipped — torch.cuda.is_available() is False]")

    print("\n" + "=" * 60)
    print("  Benchmark complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
