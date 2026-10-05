# Accent Converter

A real-time AI accent converter designed to convert Indian-accented English speech to US, UK, or Neutral English during live voice interactions.

---

## Architecture Overview

The system uses a strictly decoupled 5-stage pipeline:

```text
Audio Input ──→ Resampler ──→ Normalizer ──→ NoiseSuppressor ──→ VAD (Silero)
                                                                   │
                                                                   ▼
Enrollment Audio ──→ EcapaSpeakerEncoder (cached)         ContentEncoder (HuBERT)
                           │                                       │
                           │                                       ▼ content_rep (T, 768)
                           │                              AccentTranslator (Naive / learned)
                           │                                       │
                           │                                       ▼ target_rep (T, 768)
                           └───────────────────────────────→ SpeechSynthesizer
                                                             ┌─────────────────────────────┐
                                                             │ Acoustic Decoder / Mel Pred │
                                                             │              ↓              │
                                                             │ Neural Vocoder (HiFi-GAN)   │
                                                             └──────────────┬──────────────┘
                                                                            ▼
                                                                  Converted Waveform
```

### Canonical Audio Contract
- **Sample Rate**: 16,000 Hz
- **Channels**: 1 (mono, 1-D NumPy array)
- **Data Type**: `float32`, amplitude in `[-1.0, 1.0]`
- **Chunk Size**: 80 ms (1,280 samples)

---

## Current Status

- **Phases 1–6A Complete & Verified**:
  - **Phase 1**: Repo scaffolding & environment setup.
  - **Phase 2**: Audio foundation (polyphase resampler, normalizer, 80 ms chunker, context manager).
  - **Model ABCs**: Interface layer (`ContentEncoder`, `SpeakerEncoder`, `AccentTranslator`, `SpeechSynthesizer`) + factory.
  - **Phase 3**: Audio preprocessing (`SileroVAD` neural default, `EnergyVAD` fallback, `NoiseReduceNoiseSuppressor`, `PassthroughAEC`).
  - **Phase 4B**: HuBERT offline reference content encoder (`facebook/hubert-base-ls960`, 50 Hz, 768-dim).
  - **Phase 5**: ECAPA-TDNN speaker encoder (`speechbrain/spkrec-ecapa-voxceleb`, 192-dim embedding).
  - **Phase 6A**: Offline pipeline smoke test (`run_offline_pipeline()`, CLI runner, 5-stage plumbing verified).
- **Next Up**:
  - **Phase 6B**: Real accent conversion baseline (dataset curation, learned translator training, neural vocoder).

---

## CLI Usage

Run the offline full-utterance conversion pipeline directly from the command line:

```bash
python scripts/run_offline.py input.wav --accent us --enrollment ref.wav --output converted.wav
```

---

## Running Tests

All unit tests run offline with **zero network downloads**:

```bash
# Run all fast unit tests (285 tests)
python -m pytest tests/unit -v

# Run real model integration tests (25 tests)
python -m pytest tests/integration -v -m integration
```

---

## Documentation Index

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — System architecture, modularity rules, and directory layout.
- [`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md) — Detailed project roadmap, phase plans, and benchmark data.
- [`docs/BASELINE.md`](docs/BASELINE.md) — Baseline capabilities and empirical performance benchmarks.
- [`docs/DECISIONS.md`](docs/DECISIONS.md) — Architecture Decision Records (ADRs DEC-001 through DEC-011).
- [`docs/MODEL_SELECTION.md`](docs/MODEL_SELECTION.md) — Component selection rationale and research status.
- [`docs/MODEL_INTERFACES.md`](docs/MODEL_INTERFACES.md) — Formal interface definitions and contracts.
- [`docs/AUDIO_PIPELINE.md`](docs/AUDIO_PIPELINE.md) — Audio processing foundation and preprocessing pipeline.
