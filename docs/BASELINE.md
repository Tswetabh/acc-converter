# Baseline

## Phase 1 — Repository Stabilization (Complete)

The repository started empty. Phase 1 created the scaffolding.

- Does the repository install? **Yes** (minimal pyproject.toml).
- Does the backend start? No (unimplemented).
- Does the frontend start? No (unimplemented).
- Does model loading work? No (unimplemented).
- Can audio be captured? No (unimplemented — io.py is a stub).
- Can audio be processed? **Yes** (Phase 2 audio foundation implemented).
- Is there an existing end-to-end path? No (unimplemented).

---

## Phase 2 — Audio Foundation (Complete)

**Date**: 2026-09-20

### What was implemented

| Module | File | Status |
|--------|------|--------|
| Audio normalizer | `src/accent_converter/audio/normalizer.py` | ✅ Implemented |
| Audio resampler | `src/accent_converter/audio/resampler.py` | ✅ Implemented |
| Audio chunker | `src/accent_converter/audio/chunker.py` | ✅ Implemented |
| Streaming context | `src/accent_converter/audio/context.py` | ✅ Implemented |
| Audio package init | `src/accent_converter/audio/__init__.py` | ✅ Updated |

### Tests added

| Test file | Tests |
|-----------|-------|
| `tests/unit/test_audio_normalizer.py` | 19 tests |
| `tests/unit/test_audio_resampler.py` | 17 tests |
| `tests/unit/test_audio_chunker.py` | 22 tests |
| `tests/unit/test_audio_context.py` | 28 tests |

### Dependencies added

| Package | Version | Purpose |
|---------|---------|---------|
| scipy | installed | `resample_poly` polyphase resampling |
| pytest | installed | test runner |

numpy was already present (2.5.0). No ML frameworks installed.

### Can audio be processed?

Yes. The following pipeline is implemented and tested:

```
Raw audio (any SR/dtype/channels)
    → resample to 16 kHz
    → to_mono + to_float32 + peak_normalize
    → chunk into 80 ms frames (1 280 samples)
    → bounded look-back context (default 640 ms / 8 frames)
```

### What currently fails / is not implemented (as of Phase 2)

- `audio/io.py` — capture_audio() and play_audio() are stubs
- Backend server — not started
- Frontend — not started

---

## Phase 3 — Audio Preprocessing & Model Interfaces (Complete)

**Date**: 2026-09-21

### What was implemented

| Component | File | Status |
|---|---|---|
| Preprocessing interfaces | `src/accent_converter/audio/prep.py` | ✅ `VAD`, `NoiseSuppressor`, `AEC` ABCs + Factory |
| Energy-based VAD | `src/accent_converter/audio/prep.py` | ✅ `EnergyVAD` with hangover smoothing & dB thresholding |
| Spectral noise suppressor | `src/accent_converter/audio/prep.py` | ✅ `NoiseReduceNoiseSuppressor` wrapping `noisereduce` |
| Passthrough AEC | `src/accent_converter/audio/prep.py` | ✅ `PassthroughAEC` |
| Model ABC interfaces | `src/accent_converter/models/interfaces.py` | ✅ `ContentEncoder`, `SpeakerEncoder`, `AccentTranslator`, `StreamingVocoder` |
| Model package exports | `src/accent_converter/models/__init__.py` | ✅ Updated exports |

---

## Phase 4B — HuBERT Offline Reference Encoder (Complete)

**Date**: 2026-09-22

### Architecture Decision
HuBERT (`facebook/hubert-base-ls960`) is an **OFFLINE REFERENCE ONLY** model (`lookahead_ms = -1`). It produces linguistic/content representations intended for use in speaker/accent disentanglement. It is NOT streaming and is NOT connected to the real-time audio chunk pipeline.

### What was implemented

| Component | File | Status |
|---|---|---|
| HuBERT Content Encoder | `src/accent_converter/models/backends/content/hubert.py` | ✅ `HubertContentEncoder` using `Wav2Vec2FeatureExtractor` + `HubertModel` |
| Minimal Config Loader | `src/accent_converter/core/config.py` | ✅ `load_config(path: str) -> dict` via `yaml.safe_load` |
| Fast Unit Tests | `tests/unit/test_hubert_encoder.py` | ✅ 19 tests (0 downloads, checks inputs/unloaded guards/devices) |
| Interface Unit Tests | `tests/unit/test_model_interfaces.py` | ✅ Updated with load guards and metadata tests |
| Genuine Integration Tests | `tests/integration/test_hubert_integration.py` | ✅ 15 tests (loads checkpoint, verifies `(T, 768)` shape, determinism, actual frame counts) |
| Benchmark Script | `scripts/benchmark_hubert.py` | ✅ Empirical benchmark (CPU always, CUDA conditional) |
| Project Dependencies | `pyproject.toml` | ✅ Added `torch>=2.0`, `transformers>=4.30`, `pyyaml>=6.0` |

### Test Suite Results

```text
tests/unit:        229 passed in 7.16s (0 model downloads)
tests/integration: 15 passed in 51.79s (loads real facebook/hubert-base-ls960)
Total:             244 passed
```

### Empirical Benchmark Measurements (Phase 4B Measured)

Obtained via `python scripts/benchmark_hubert.py`:

- **Hardware**: NVIDIA GeForce RTX 3050 6GB Laptop GPU (6.00 GB total VRAM)
- **Host OS**: Windows 11, Python 3.14.5, `torch==2.14.0+cu126`, `transformers==5.17.0`
- **Model**: `facebook/hubert-base-ls960` (94.68 M params, output dim 768)
- **Model Load Time**:
  - CPU: **7.394 s**
  - CUDA: **2.900 s**
- **Peak CUDA VRAM**: **450.6 MB**

#### Latency, Frame Counts & Real-Time Factor (RTF)

| Duration | Measured Frames | Frame Rate (obs) | CPU Latency | CPU RTF | CUDA Latency | CUDA RTF |
|:---|:---|:---|:---|:---|:---|:---|
| **1.0 s** | 49 | 49.00 Hz | 60.01 ms | 0.0600 | 11.10 ms | 0.0111 |
| **3.0 s** | 149 | 49.67 Hz | 144.49 ms | 0.0482 | 18.78 ms | 0.0063 |
| **5.0 s** | 249 | 49.80 Hz | 225.17 ms | 0.0450 | 25.45 ms | 0.0051 |

---

## Phase 5 — ECAPA Speaker Encoder (Complete)

**Date**: 2026-09-23

### What was implemented

| Component | File | Status |
|---|---|---|
| ECAPA Speaker Encoder | `src/accent_converter/models/backends/speaker/ecapa.py` | ✅ `EcapaSpeakerEncoder` using SpeechBrain `EncoderClassifier` |
| Fast Unit Tests | `tests/unit/test_ecapa_encoder.py` | ✅ 20 tests (0 downloads, mocked classifier, input guards, 192-dim contract) |
| Integration Tests | `tests/integration/test_ecapa_integration.py` | ✅ 5 tests (real model, (192,) float32 output, determinism, cosine similarity) |
| Dependencies Added | `pyproject.toml` | ✅ Added `speechbrain>=1.0` |

---

## Preprocessing Upgrade — Silero Neural VAD (Complete)

**Date**: 2026-09-24

### What was implemented

| Component | File | Status |
|---|---|---|
| Silero VAD Backend | `src/accent_converter/audio/preprocessing/vad/silero.py` | ✅ `SileroVAD` (neural, streaming 512-window adapter, hangover smoothing) |
| Factory Registration | `src/accent_converter/audio/preprocessing/factory.py` | ✅ Registered under `backend: "silero"` |
| Configuration | `configs/default_config.yaml` | ✅ Set `backend: silero` (default) |
| Fast Unit Tests | `tests/unit/test_silero_vad.py` | ✅ 22 tests (0 downloads, input guards, timing preservation, hangover, buffer) |
| Dependencies Added | `pyproject.toml` | ✅ Added `silero-vad>=6.0` |

### Test Suite Results

```text
tests/unit:        273 passed in 17.33s (0 model downloads)
tests/integration: 20 passed (15 HuBERT + 5 ECAPA)
Total:             293 passed
```

---

## Phase 6A — Offline Pipeline Smoke Test (Complete)

**Date**: 2026-09-25

### What was implemented

| Component | File | Status |
|---|---|---|
| Offline Conversion Pipeline | `src/accent_converter/pipeline/offline.py` | ✅ `run_offline_pipeline()` wires preprocessing, ECAPA, HuBERT, Naive translator, Vocoder stub |
| Vocoder Stub Mode | `src/accent_converter/models/backends/synthesis/vocoder.py` | ✅ Stub mode (`checkpoint=None`), validates shapes `(T, D)` and `(192,)`, returns $T \times 320$ zeros |
| Pipeline Package Init | `src/accent_converter/pipeline/__init__.py` | ✅ Exports `run_offline_pipeline` |
| CLI Runner | `scripts/run_offline.py` | ✅ Argument parser supporting `--accent`, `--enrollment`, `--output`, `--config`, `--device` |
| Fast Unit Tests | `tests/unit/test_offline_pipeline.py` | ✅ 10 tests (mocked models, audio validation, format handling, zero network) |
| Model Interface Tests | `tests/unit/test_model_interfaces.py` | ✅ Updated for Vocoder stub mode and contract verification |
| End-to-End Integration Test | `tests/integration/test_offline_pipeline.py` | ✅ 1 test using real HuBERT + ECAPA weights (asserts duration matching $\pm 10\%$, finite float32) |
| Windows Symlink Resilience | `src/accent_converter/models/backends/speaker/ecapa.py` | ✅ Uses `LocalStrategy.COPY` on Windows to eliminate `WinError 1314` symlink permission errors |

### Test Suite Results (Post-Phase 6A)

```text
tests/unit:        285 passed in 21.27s (0 model downloads)
tests/integration:  25 passed in 35.03s (15 HuBERT + 9 ECAPA + 1 Offline Pipeline)
Total:             310 passed
```

---

## Python Environment

```text
Python:       3.14.5
numpy:        2.5.0
scipy:        1.18.1
torch:        2.14.0+cu126
CUDA:         Available (NVIDIA GeForce RTX 3050 6GB Laptop GPU)
transformers: 5.17.0
speechbrain:  1.1.1
silero-vad:   6.2.3
pyyaml:       6.0.3
pytest:       9.1.1
```
