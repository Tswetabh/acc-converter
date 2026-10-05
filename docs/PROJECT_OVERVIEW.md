# Accent Converter — Project Overview

> **Last updated**: 2026-09-25 (after Phase 6A completion)
>
> This is the single authoritative reference for what we are building,
> how we are building it, what is done, and what remains.

---

## 1. Project Goal

Build a **real-time AI accent converter** that converts:

> **Indian English speech → US / UK / Neutral English speech**

during a live browser voice interaction, with low enough latency that the
conversation feels natural to both participants.

The system must be **streaming** — it cannot record a complete utterance,
process it offline, and play back the result with a multi-second delay. Each
frame of audio arriving from the microphone must yield converted audio with
bounded latency.

---

## 2. Final Pipeline (Target System)

```
Browser Microphone
        |
        v  (WebRTC / WebSocket transport)
+-----------------------------------------+
|  AUDIO FOUNDATION  (Phase 2 complete)   |
|                                         |
|  Resampler       --> 16 kHz             |
|  Normalizer      --> mono float32 [-1,1]|
|  Chunker         --> 80 ms frames       |
|  ContextManager  --> 640 ms lookback    |
+-----------------------------------------+
        |
        v
+------------------------------------------+
|  AUDIO PREPROCESSING  (Phase 3 complete) |
|                                          |
|  AEC               --> PassthroughAEC    |  real AEC via browser WebRTC (Phase 11)
|  Noise Suppressor  --> noisereduce       |  spectral gating
|  VAD               --> SileroVAD         |  neural speech detection + hangover (EnergyVAD fallback)
+------------------------------------------+
        |  (speech frames only)
        v
+----------------------------------------------+
|  CONTENT ENCODER  (Phase 4B - implemented)   |
|                                              |
|  HuBERT Base (offline reference only)        |  --> (T, 768) float32
|    lookahead_ms = -1  (non-causal)           |  NOT streaming
|                                              |
|  TVTSyn Causal Encoder (Phase 7, TBD)        |  --> (T, D) float32, streaming
|    lookahead_ms <= 80 ms, ring KV-cache      |  BUILD FROM SCRATCH
+----------------------------------------------+
        |  linguistic/content representation
        v
+------------------------------------------+
|  ACCENT TRANSLATOR  (Phase 6 - pending)  |
|                                          |
|  NaiveAccentTranslator (passthrough)     |  <-- current stub (identity fn)
|  Real Translator (Phase 6, TBD)          |  seq2seq / non-parallel CTC / PPG
+------------------------------------------+
        |  accent-converted representation
        |          ^
        |   +-----------------------------------+
        |   |  SPEAKER ENCODER  (Phase 5 next)  |
        |   |                                   |
        |   |  ECAPA-TDNN (speechbrain)         |  --> (192,) float32 speaker emb
        |   |  Computed once from enrollment    |
        |   +-----------------------------------+
        v
+----------------------------------------------+
|  WAVEFORM SYNTHESIZER  (Phase 8 - pending)   |
|                                              |
|  Vocoder (HiFi-GAN / UnivNet / BigVGAN)      |  <-- primary (senior project req)
|  TVTSyn Decoder                              |  <-- alternative
+----------------------------------------------+
        |  converted audio waveform (16 kHz float32)
        v
+--------------------------------------+
|  OUTPUT STITCHING  (Phase 9 - TBD)   |
|  overlap-add / crossfade             |
+--------------------------------------+
        |
        v  (WebRTC / WebSocket transport)
Browser Playback / Remote Participant
```

---

## 3. Implementation Status

### Complete

| Phase | What Was Built | Tests | Key Files |
|---|---|---|---|
| **1** — Repo Stabilization | Empty repo scaffolded, package structure, `pyproject.toml` | — | `src/accent_converter/`, `conftest.py` |
| **2** — Audio Foundation | Resampler, Normalizer, Chunker, AudioContextManager | **86 unit tests** | `audio/resampler.py`, `normalizer.py`, `chunker.py`, `context.py` |
| **Model ABCs** | `ContentEncoder`, `SpeakerEncoder`, `AccentTranslator`, `SpeechSynthesizer` interfaces + factory | Included in interface tests | `models/interfaces.py`, `models/factory.py` |
| **3** — Audio Preprocessing | SileroVAD (neural default), EnergyVAD (fallback), NoiseReduceNoiseSuppressor, PassthroughAEC + ABC + factory | **87 unit tests** | `audio/preprocessing/` |
| **4B** — HuBERT Offline Reference | `HubertContentEncoder.load()` + `encode()` using `Wav2Vec2FeatureExtractor` + `HubertModel` | **19 unit + 15 integration** | `models/backends/content/hubert.py` |
| **5** — ECAPA Speaker Encoder | `EcapaSpeakerEncoder.load()` + `encode()` using SpeechBrain ECAPA-TDNN `(192,)` embedding | **20 unit + 9 integration** | `models/backends/speaker/ecapa.py` |
| **6A** — Offline Pipeline Smoke Test | `run_offline_pipeline()` end-to-end wiring of all 5 canonical components + CLI (`scripts/run_offline.py`) | **10 unit + 1 integration** | `pipeline/offline.py`, `scripts/run_offline.py` |

### In Progress

None (Ready for Phase 6B)

### Not Started

| Phase | Description | Blocks on |
|---|---|---|
| **6B** — Real Conversion Baseline | Train/integrate real translator + acoustic decoder + HiFi-GAN vocoder | Phase 6A verified |
| **7** — Streaming Causal Content Encoder | Build TVTSyn-style causal CNN + MHSA encoder from scratch | Phase 6B working |
| **8** — Causal Waveform Generation | Integrate HiFi-GAN / UnivNet / BigVGAN vocoder | Phase 6B working |
| **9** — Output Stitching | Overlap-add / crossfade between chunks | Phase 8 |
| **10** — Real-time Concurrent Pipeline | Concurrent queues, audio ring buffers, threading/asyncio | Phase 8 |
| **11** — Browser Integration | WebRTC/WebSocket transport, browser mic capture | Phase 10 |

---

## 4. Test Suite Summary

```text
Command                                                  Tests   Time      Downloads
---------------------------------------------------------------------------------------
pytest tests/unit -v                                     285     ~21 s     0 (none)
pytest tests/integration -v -m integration                25     ~35 s     0 (cached)
---------------------------------------------------------------------------------------
Total                                                    310     ~56 s
```

| Test File | Count | Description |
|---|---|---|
| `test_audio_normalizer.py` | 20 | Normalizer — dtype handling, peak norm, mono |
| `test_audio_resampler.py` | 17 | Resampler — all common rates, edge cases |
| `test_audio_chunker.py` | 22 | Chunker — 80 ms frames, partial frames |
| `test_audio_context.py` | 28 | ContextManager — sliding window, lookback |
| `test_preprocessing_interfaces.py` | 65 | VAD, NoiseSuppressor, AEC, factory |
| `test_silero_vad.py` | 22 | Silero VAD — neural inference, hangover, streaming buffer |
| `test_model_interfaces.py` | 42 | Model ABCs, factory, stubs, metadata, vocoder stub |
| `test_hubert_encoder.py` | 19 | HuBERT unit (no download) — input guards, validation |
| `test_ecapa_encoder.py` | 20 | ECAPA unit (no download) — input guards, 192-dim contract |
| `test_offline_pipeline.py` (unit) | 10 | Offline pipeline unit — stages, validation, mocking |
| `test_hubert_integration.py` | 15 | HuBERT real model — shape, determinism, frames |
| `test_ecapa_integration.py` | 9 | ECAPA real model — shape, determinism, similarity, latency |
| `test_offline_pipeline.py` (integration) | 1 | End-to-end full utterance with real weights |

---

## 5. Canonical Audio Format

Every module in the pipeline passes audio in this exact format:

| Property | Value |
|---|---|
| **Sample rate** | 16 000 Hz |
| **Channels** | 1 (mono, 1-D NumPy array) |
| **dtype** | `np.float32` |
| **Amplitude range** | approx [-1, 1] |
| **Chunk size** | 80 ms = 1 280 samples |
| **Lookback** | 640 ms = 8 frames (default) |

---

## 6. Model Interchangeability

The entire model layer is built on **abstract base classes (ABCs)** in
[`models/interfaces.py`](file:///d:/Accent%20Converter/src/accent_converter/models/interfaces.py).
All pipeline code touches only interfaces — never concrete backends. Swapping
a model requires only a YAML config change and registering the new class in
[`factory.py`](file:///d:/Accent%20Converter/src/accent_converter/models/factory.py).

```yaml
# Switching content encoder from HuBERT to a future causal encoder:
models:
  content_encoder:
    backend: tvtsyn_causal   # was: hubert
    checkpoint: path/to/causal_encoder.ckpt
```

### 6.1 Content Encoder — Interchangeable Backends

| Backend | Config Key | Status | `lookahead_ms` | `output_dim` | Notes |
|---|---|---|---|---|---|
| **HuBERT Base** (`facebook/hubert-base-ls960`) | `hubert` | **Implemented** | `-1` (offline ref) | 768 | Non-causal; offline use only |
| **HuBERT Large** | (custom) | Not started | `-1` | 1024 | Same pattern, higher quality |
| **WavLM Base+** (`microsoft/wavlm-base-plus`) | (custom) | Not started | `-1` | 768 | Better speaker disentanglement |
| **TVTSyn Causal Encoder** | `tvtsyn_causal` | Phase 7 (build from scratch) | `<= 80` | TBD | Streaming; no public checkpoint |

> [!IMPORTANT]
> **Two-Tier Architecture (DEC-010)**:
> - Phases 4–6: HuBERT offline reference encoder for development & validation.
> - Phase 7+: Custom causal streaming encoder replaces HuBERT in the live path.
> - HuBERT is **never** used in the live streaming pipeline.

### 6.2 Speaker Encoder — Interchangeable Backends

| Backend | Config Key | Status | `output_dim` | Notes |
|---|---|---|---|---|
| **ECAPA-TDNN** (`speechbrain/spkrec-ecapa-voxceleb`) | `ecapa` | Phase 5 (next) | 192 | Industry standard; Apache 2.0 |
| **TitaNet** (NVIDIA NeMo) | (custom) | Not started | 192–256 | Alternative if ECAPA unavailable |
| **d-vector** (Resemblyzer) | (custom) | Not started | 256 | Lighter weight option |

Speaker embedding is computed **once** from an enrollment/reference audio clip,
then cached by the caller for the duration of the session.

### 6.3 Accent Translator — Interchangeable Backends

| Backend | Config Key | Status | Notes |
|---|---|---|---|
| **NaiveAccentTranslator** | `naive` | **Implemented** (passthrough) | Identity — returns content rep unchanged |
| **Seq2Seq Translator** | TBD | Phase 6 (research first) | Non-parallel accent translation |
| **PPG-based Translator** | TBD | Phase 6 | Phonetic Posteriorgrams approach |
| **Non-parallel CTC** | TBD | Phase 6 | CTC-constrained approach |

### 6.4 Waveform Synthesizer — Interchangeable Backends

| Backend | Config Key | Status | Notes |
|---|---|---|---|
| **VocoderSynthesizer** | `vocoder` | Phase 8 stub | HiFi-GAN / UnivNet / BigVGAN (TBD) |
| **TVTSynSynthesizer** | `tvtsyn` | Phase 8 stub | From TVTSyn paper — streaming-compatible |

### 6.5 Audio Preprocessing — Interchangeable Backends

| Component | Current Backend | Config Key | Status | Possible Alternatives |
|---|---|---|---|---|
| VAD | `SileroVAD` | `silero` | **Implemented (Default)** | EnergyVAD (fallback), WebRTC VAD (browser Phase 11) |
| Noise Suppression | `NoiseReduceNoiseSuppressor` | `noisereduce` | **Implemented** | DeepFilterNet (Phase 4+), RNNoise |
| AEC | `PassthroughAEC` | `passthrough` | **Wired** | Browser WebRTC AEC3 (Phase 11), SpeexAEC |

---

## 7. Detailed Phase Plan

### Phase 1 — Repository Stabilization (Complete)

- Empty repo audited, package structure created.
- `src/accent_converter/` layout established with `pyproject.toml`.
- Baseline recorded in [`docs/BASELINE.md`](file:///d:/Accent%20Converter/docs/BASELINE.md).

---

### Phase 2 — Audio Foundation (Complete)

**Goal**: Accept any raw audio → canonical 16 kHz mono float32 chunks.

```
Raw audio (any SR, dtype, channels)
  --> resample to 16 kHz   (scipy.signal.resample_poly, polyphase rational)
  --> to_mono + to_float32 + peak_normalize
  --> chunk into 80 ms frames (1 280 samples)
  --> bounded look-back context (640 ms / 8 frames default)
```

Key design decisions:
- Polyphase rational resampling (no aliasing artefacts, exact integer ratios).
- Silent frames not rescaled (avoids amplifying the noise floor).
- Context manager uses a deque — O(1) append/pop, bounded memory.

---

### Phase 3 — Audio Preprocessing (Complete)

**Goal**: Gate model inference with VAD; suppress background noise; wire the AEC slot.

| Component | Implementation | Design Note |
|---|---|---|
| `SileroVAD` | Pre-trained neural VAD, streaming window adapter + hangover counter | Default backend. `threshold=0.5`, `hangover_frames=8`, noise-robust |
| `EnergyVAD` | Pure NumPy, short-time energy + hangover counter | Algorithmic fallback. `threshold_db=-40`, `hangover_frames=8` configurable |
| `NoiseReduceNoiseSuppressor` | `noisereduce.reduce_noise` spectral gating | Operates on full buffer per call; use post-VAD on complete segments for best results |
| `PassthroughAEC` | Identity — passes near-end audio unchanged | Real AEC done browser-side (WebRTC AEC3) in Phase 11 |

---

### Phase 4B — HuBERT Offline Reference Encoder (Complete)

**Goal**: Implement and verify the `ContentEncoder` interface end-to-end with a real model.

What was implemented:
- `HubertContentEncoder.load(checkpoint_path, device)` using explicit `Wav2Vec2FeatureExtractor` + `HubertModel`.
- `encode(audio: np.ndarray) --> np.ndarray` — full input validation, `torch.inference_mode()`, returns `(T, 768) float32`.
- Auto-selects CUDA or CPU; allows explicit device override.
- Metadata properties: `frame_rate_hz=50`, `output_dim=768`, `lookahead_ms=-1`.
- Minimal `load_config(path) --> dict` in `core/config.py`.

Empirical measurements on NVIDIA RTX 3050 6GB (Windows, Python 3.14.5):

| Metric | CPU | CUDA |
|---|---|---|
| Model load time | 7.394 s | 2.900 s |
| Latency — 1 s audio | 60.01 ms | 11.10 ms |
| Latency — 3 s audio | 144.49 ms | 18.78 ms |
| Latency — 5 s audio | 225.17 ms | 25.45 ms |
| Peak VRAM | — | 450.6 MB |
| Actual frames (1 s) | 49 | 49 |
| Actual frames (3 s) | 149 | 149 |
| Actual frames (5 s) | 249 | 249 |

Frame count formula: `T = floor((N - 400) / 320) + 1`
(7 conv layers, cumulative stride 320 samples = 20 ms, receptive field 400 samples = 25 ms)

---

### Phase 5 — ECAPA Speaker Encoder (Complete)

**Goal**: Compute a 192-dim speaker embedding from an enrollment utterance.

What was implemented:
- Integrated `speechbrain>=1.0.0` and `torchaudio`.
- `EcapaSpeakerEncoder.load()` via `speechbrain.inference.classifiers.EncoderClassifier`.
- `encode(reference_audio) --> np.ndarray` shape `(192,)`, `float32`.
- Automatic device fallback (`cuda` if available, else `cpu`).
- Stateless across chunks (`reset()` is safe no-op).
- Enforces strict input validation (non-empty 1-D float32, NaN/Inf rejection).
- Exposes `@property embedding_dim -> int = 192`.
- Fully verified with 20 unit tests (mocked, 0 downloads) and 5 integration tests (real model).

---

### Phase 6A — Offline Pipeline Smoke Test (Complete)

**Date**: 2026-09-25

**Goal**: Wire the full offline pipeline and produce a converted WAV file.

```
enrollment_audio.wav
  --> ECAPA speaker embedding (192-dim)

source_audio.wav
  --> Audio preprocessing (resample to 16 kHz mono, peak norm, noise suppression)
  --> HuBERT content encoder (T, 768)
  --> NaiveAccentTranslator (passthrough)
  --> VocoderSynthesizer (Phase 6A duration-matching zeros stub)
  --> converted_audio.wav (optional WAV output)
```

- **Plumbing Verified**: Verified all 5 canonical components connected in `run_offline_pipeline()` (`src/accent_converter/pipeline/offline.py`).
- **CLI Available**: `python scripts/run_offline.py input.wav --accent us --enrollment ref.wav --output out.wav`.
- **Test Coverage**: 10 isolated unit tests (`tests/unit/test_offline_pipeline.py`) + 1 end-to-end integration test with real weights (`tests/integration/test_offline_pipeline.py`).
- **Duration Matching**: Verified $\approx 1.5$ s input audio produces 73 HuBERT frames $\to 73 \times 320 = 23,360$ samples ($\approx 1.46$ s, well within $\pm 10\%$).

---

### Phase 6B — Real Accent Conversion Baseline (Next Up)

**Goal**: Replace the passthrough translator and stub vocoder with learned conversion models.
- **Dataset Preparation**: Curate L2-ARCTIC (Indian English speakers) + VCTK (US/UK reference speakers).
- **Phonetic/Content Representation**: Freeze layer representation (e.g. HuBERT Layer 9 vs PPGs).
- **Learned Accent Translator**: Train/evaluate seq2seq or non-parallel translator model.
- **Acoustic Decoder + Vocoder**: Implement acoustic bridge to mel spectrogram + pre-trained HiFi-GAN.

---

### Phase 7 — Streaming Causal Content Encoder (Not Started)

**Goal**: Replace HuBERT (offline) with a genuinely streaming causal encoder.

Architecture target:
- Causal 1-D CNN front-end.
- 8x causal Multi-Head Self-Attention (MHSA) blocks with ring KV-cache (2 s window).
- VQ bottleneck (8-dim codebook) for speaker disentanglement.
- `lookahead_ms <= 80`.
- Must be **trained from scratch** — no public checkpoint exists.

This is a significant research task. HuBERT remains the offline reference throughout.

---

### Phase 8 — Causal Waveform Generation (Not Started)

**Goal**: Integrate a streaming neural vocoder.

Primary candidates:
- **HiFi-GAN** — excellent quality; check streaming / chunk-by-chunk support.
- **UnivNet** — lightweight; may support chunk inference.
- **BigVGAN** — highest quality; heavier VRAM.

Alternative: TVTSyn decoder (same architecture paper as the Phase 7 causal encoder).

---

### Phase 9 — Output Stitching (Not Started)

Overlap-add / crossfade to eliminate chunk boundary discontinuities.
Measure: clicks, phase issues, volume modulation, duplicated audio.

---

### Phase 10 — Real-Time Concurrent Pipeline (Not Started)

Replace sequential processing with concurrent queues:
`audio capture || preprocessing || inference || playback`

Use Python `asyncio` / `threading` / ring buffers. Avoid blocking the audio
callback on expensive ML inference.

---

### Phase 11 — Browser Integration (Not Started)

WebRTC/WebSocket transport, browser mic capture via `getUserMedia`,
browser-side WebRTC AEC3, low-latency playback.

---

## 8. Environment & Dependencies

```text
Python:       3.14.5 (Windows)
numpy:        2.5.0
scipy:        1.18.1       (polyphase resampler)
noisereduce:  3.x          (spectral gating NS)
torch:        2.14.0+cu126 (ML inference engine)
CUDA:         Available    (NVIDIA GeForce RTX 3050 6GB Laptop GPU)
transformers: 5.17.0       (HuBERT via HuggingFace)
pyyaml:       6.0.3        (YAML config loading)
pytest:       9.1.1        (test runner)

Pending for Phase 5:
  speechbrain>=1.0.0       (ECAPA-TDNN speaker encoder)
  torchaudio>=2.1.0        (audio loading for SpeechBrain)
```

---

## 9. Repository Structure

```
d:/Accent Converter/
├── src/accent_converter/
│   ├── audio/
│   │   ├── resampler.py           Phase 2 done
│   │   ├── normalizer.py          Phase 2 done
│   │   ├── chunker.py             Phase 2 done
│   │   ├── context.py             Phase 2 done
│   │   └── preprocessing/         Phase 3 done
│   │       ├── interfaces.py        (VAD, NS, AEC ABCs + factory)
│   │       ├── vad/energy.py        (EnergyVAD)
│   │       ├── noise/               (NoiseReduceNoiseSuppressor)
│   │       └── aec/passthrough.py   (PassthroughAEC)
│   ├── core/
│   │   └── config.py              Phase 4B done (load_config via yaml.safe_load)
│   └── models/
│       ├── interfaces.py          ContentEncoder, SpeakerEncoder,
│       │                          AccentTranslator, SpeechSynthesizer ABCs
│       ├── factory.py             build_*() functions (deferred imports)
│       └── backends/
│           ├── content/
│           │   └── hubert.py      HubertContentEncoder  Phase 4B done
│           ├── speaker/
│           │   └── ecapa.py       EcapaSpeakerEncoder   Phase 5 stub --> impl
│           ├── accent/
│           │   └── naive.py       NaiveAccentTranslator done (passthrough)
│           └── synthesis/
│               ├── vocoder.py     VocoderSynthesizer    Phase 8 stub
│               └── tvtsyn.py     TVTSynSynthesizer     Phase 8 stub
├── tests/
│   ├── unit/                      229 tests, 0 model downloads
│   │   ├── test_audio_*.py          (Phase 2: 86 tests)
│   │   ├── test_preprocessing_interfaces.py  (Phase 3: 84 tests)
│   │   ├── test_model_interfaces.py          (Model ABCs: 40 tests)
│   │   └── test_hubert_encoder.py            (Phase 4B: 19 tests)
│   └── integration/               Real model tests (separate from unit)
│       └── test_hubert_integration.py   (Phase 4B: 15 tests)
├── scripts/
│   ├── benchmark_hubert.py        Phase 4B empirical measurements
│   └── benchmark_ecapa.py         Phase 5 (pending)
├── docs/
│   ├── PROJECT_OVERVIEW.md        THIS FILE
│   ├── ARCHITECTURE.md            Pipeline diagram + phase table
│   ├── BASELINE.md                Empirical measurements per phase
│   ├── CONTENT_ENCODER.md         HuBERT research + benchmarks
│   ├── MODEL_SELECTION.md         Selection decisions per component
│   ├── DECISIONS.md               DEC-001...DEC-010 architectural decisions
│   └── AUDIO_PIPELINE.md          Audio foundation detailed docs
├── configs/                       YAML configuration (to be built out)
├── pytest.ini                     testpaths = tests/unit (integration isolated)
└── pyproject.toml
```

---

## 10. Key Architectural Decisions (Summary)

| # | Decision | Summary |
|---|---|---|
| DEC-001 | ABCs for all model interfaces | Pipeline depends only on interfaces, not concrete backends |
| DEC-002 | Deferred imports in `factory.py` | ML frameworks not loaded until `build_*()` is called |
| DEC-003 | Polyphase rational resampling | `scipy.signal.resample_poly` — no aliasing |
| DEC-004 | Named factory functions | `build_content_encoder()`, not a class-based registry |
| DEC-005 | `NaiveAccentTranslator` passthrough stub | Wires slot until Phase 6 |
| DEC-006 | `EnergyVAD` instead of `webrtcvad` | Pure NumPy, works on Python 3.14 Windows (no MSVC needed) |
| DEC-007 | `noisereduce` for noise suppression | No C extension, Python 3.14 compatible |
| DEC-008 | `PassthroughAEC` with `far_end` parameter | Real AEC via browser WebRTC in Phase 11 |
| DEC-009 | Preprocessing separate from `models/` | Different lifecycle: no `load(checkpoint)` needed |
| DEC-010 | Two-tier content encoder strategy | HuBERT offline reference (Phases 4-6) + TVTSyn causal streaming (Phase 7) |

---

## 11. Latency Budget (Empirical — In Progress)

> [!WARNING]
> No latency targets are set until measured. Numbers below are Phase 4B only.

| Stage | Measured | Target |
|---|---|---|
| Resampler + Normalizer + Chunker | Not yet measured | < 1 ms per chunk |
| EnergyVAD | Not yet measured | < 0.5 ms per chunk |
| NoiseReduceNoiseSuppressor | Not yet measured | < 5 ms per chunk |
| HuBERT content encoder — CUDA, 1 s | **11.1 ms** | N/A (offline reference) |
| HuBERT content encoder — CPU, 1 s | **60.0 ms** | N/A (offline reference) |
| TVTSyn causal encoder (streaming) | Not built | <= 80 ms per chunk |
| ECAPA speaker encoder (enrollment) | Not yet measured | < 100 ms (one-time) |
| Accent Translator | Not implemented | TBD |
| Vocoder (chunk-by-chunk) | Not implemented | TBD |
| **Total end-to-end (streaming target)** | — | **< 200 ms** |

---

## 12. Quality Goals (To Be Evaluated After Phase 6)

| Metric | Method |
|---|---|
| Content preservation | WER comparison: source vs. converted (ASR) |
| Accent reduction | Accent classifier + human evaluation |
| Speaker similarity | Cosine similarity of ECAPA embeddings |
| Speech naturalness | UTMOS / MOS objective metric + human listening tests |
| Streaming RTF | Empirical per-stage timing (Phase 10) |
| Session stability | 10-min streaming test, check dropout / NaN / queue overflow |

---

## Update — 2026-10-05

* **Phase 6B complete:** Acoustic Decoder trained (50 utterances, val L1 1.41), HiFi-GAN vocoder integrated, end-to-end synthesis verified.
* **Phase 6C in progress:** L2-ARCTIC + CMU ARCTIC extracted, parallel manifests (3,848 train / 224 val / 1,583 test) built, DTW alignment engine implemented.
* **Next:** full DTW run, decoder retrain on full data, Conformer Accent Translator (6D).


* **Phase 6C Complete:** Full DTW alignment executed on all 3,848 training pairs (stored in `data/aligned/arctic_train`).
* **Phase 6D Complete:** Built and trained `ConformerAccentTranslator` (6-layer Conformer on 50 Hz continuous HuBERT features). Validation L1 dropped from 0.2273 baseline to 0.1609. Checkpoint saved at `checkpoints/accent_translator/best.pt`.

