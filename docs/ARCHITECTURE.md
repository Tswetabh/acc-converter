# Architecture

> **Phases 1–6A complete** (Audio foundation, Preprocessing with Silero VAD, HuBERT content encoder, ECAPA speaker encoder, Offline pipeline smoke test).
> **Current status**: Phase 6A complete. Ready for Phase 6B (learned accent translation & acoustic decoder).

---

## System Overview

```
Microphone / Browser Audio
        │
        ▼
  ┌─────────────────────┐
  │   Audio Foundation  │  Phase 2 ✅
  │                     │
  │  Resampler          │  → 16 kHz
  │  Normalizer         │  → mono, float32, [-1,1]
  │  Chunker            │  → 80 ms frames (1 280 samples)
  │  AudioContextManager│  → bounded look-back cache
  │                     │
  │  Audio Preprocessing│  Phase 3 ✅
  │  - SileroVAD        │  → neural speech detection + hangover (EnergyVAD fallback)
  │  - NoiseSuppressor  │  → noisereduce spectral gating
  │  - PassthroughAEC   │  → timing preserved; browser AEC in Phase 11
  └──────────┬──────────┘
             │ (speech frames)
             ▼
  ┌──────────────────────┐
  │  ContentEncoder      │  Phase 4B ✅ (HuBERT Base, 50 Hz, (T, 768))
  │                      │  Phase 7 ⬜ (Causal TVTSyn encoder)
  └──────────┬───────────┘
             │ content_rep
             ▼
  ┌──────────────────────┐
  │  AccentTranslator    │  Phase 6A ✅ (Naive passthrough stub)
  │  (+ target accent)   │  Phase 6B ⬜ (Learned translation network)
  └──────────┬───────────┘
             │ target_rep
             ▼
  ┌──────────────────────┐   Enrollment Audio
  │  SpeechSynthesizer   │          │
  │  ┌────────────────┐  │          ▼
  │  │Acoustic Decoder│  │   ┌───────────────┐
  │  └───────┬────────┘  │   │SpeakerEncoder │  Phase 5 ✅ (ECAPA-TDNN, (192,))
  │          ▼ (mel)     │   └───────┬───────┘
  │  ┌────────────────┐  │           │ speaker_embedding
  │  │ Neural Vocoder │  │◄──────────┘ (Phase 6A ✅ stub; Phase 8 ⬜ vocoder)
  │  └────────────────┘  │
  └──────────┬───────────┘
             │
             ▼
  Output Audio (16 kHz, float32)
             │
             ▼
  WebRTC / WebSocket transport  →  Browser  (Phase 11)
```

---

## Modularity Principle

**The pipeline code depends only on interfaces, never on concrete backends.**

```python
# ✅ Correct — pipeline imports only the interface and the factory
from accent_converter.models import SpeechSynthesizer, build_synthesizer

synthesizer: SpeechSynthesizer = build_synthesizer(config["synthesis"])
waveform = synthesizer.synthesize(converted_rep, speaker_embedding)

# ❌ Wrong — pipeline must never import a concrete backend directly
from accent_converter.models.backends.synthesis.vocoder import VocoderSynthesizer
```

Swapping from vocoder to TVTSyn requires only a config change:

```yaml
synthesis:
  backend: tvtsyn       # was: vocoder
  checkpoint: path/to/tvtsyn.ckpt
```

---

## Directory Structure (models layer)

```
src/accent_converter/models/
├── interfaces.py              ← ABCs (ContentEncoder, SpeakerEncoder,
│                                       AccentTranslator, SpeechSynthesizer)
├── factory.py                 ← build_synthesizer(), build_content_encoder(), …
├── backends/
│   ├── synthesis/
│   │   ├── vocoder.py         ← VocoderSynthesizer (stub, Phase 8)
│   │   └── tvtsyn.py          ← TVTSynSynthesizer  (stub, Phase 8)
│   ├── content/
│   │   └── hubert.py          ← HubertContentEncoder (implemented, Phase 4B)
│   ├── speaker/
│   │   └── ecapa.py           ← EcapaSpeakerEncoder (implemented, Phase 5)
│   └── accent/
│       └── naive.py           ← NaiveAccentTranslator (passthrough, Phase 6A)
└── __init__.py                ← re-exports interfaces + factory functions
```

---

## Audio Layer Directory Structure

```
src/accent_converter/audio/
├── normalizer.py         ← to_mono, to_float32, peak_normalize, normalize
├── resampler.py          ← resample (polyphase rational, scipy)
├── chunker.py            ← chunk_audio, frame_generator (80 ms / 1 280 samples)
├── context.py            ← AudioContextManager (bounded sliding window)
├── prep.py               ← redirect shim → preprocessing/
├── preprocessing/        ← Phase 3 audio preprocessing
│   ├── interfaces.py     ← VoiceActivityDetector, NoiseSuppressor, AcousticEchoCanceller ABCs
│   ├── factory.py        ← build_vad(), build_noise_suppressor(), build_aec()
│   ├── vad/
│   │   ├── silero.py     ← SileroVAD (neural VAD, default)
│   │   └── energy.py     ← EnergyVAD (pure numpy fallback, threshold + hangover)
│   ├── noise/
│   │   └── noisereduce_backend.py  ← NoiseReduceNoiseSuppressor (spectral gating)
│   └── aec/
│       └── passthrough.py ← PassthroughAEC (wires slot; real AEC via browser Phase 11)
└── __init__.py           ← public re-exports
```

---

## Implementation Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Repository stabilization | ✅ Complete |
| 2 | Audio foundation | ✅ Complete |
| — | Model interfaces & abstractions | ✅ Complete |
| 3 | Audio preprocessing (VAD, NS, AEC stub) | ✅ Complete |
| 4 | Content encoder (HuBERT offline ref) | ✅ Complete |
| 5 | Speaker encoder (ECAPA-TDNN) | ✅ Complete |
| 6A | Offline pipeline smoke test | ✅ Complete |
| 6B | Real acoustic decoder & HiFi-GAN baseline | ✅ Complete |
| 6C | Dataset curation & full DTW alignment (3,848 pairs) | ✅ Complete |
| 6D | Conformer Accent Translator training | ✅ Complete |
| 7 | Streaming content encoder (Causal TVTSyn / chunked HuBERT) | ⬜ Next |
| 8 | Causal speech synthesis backend | ⬜ Pending Phase 7 |
| 9 | Streaming accent translator inference | ⬜ Pending Phase 7 |
| 10 | Real-time concurrent audio engine (ring buffers, queues) | ⬜ Pending Phase 8 |
| 11 | Browser Web Studio & WebRTC transport | 🔄 In Progress (Studio UI live) |

---

## Canonical Audio Format

| Property | Value |
|----------|-------|
| Sample rate | 16 000 Hz |
| Channels | mono (1-D) |
| dtype | float32 |
| Amplitude | ≈ [−1, 1] |
| Chunk size | 80 ms = 1 280 samples |
> **Update 2026-10-05:** Phase 6B complete (Acoustic Decoder + HiFi-GAN synthesis verified). Phase 6C (data prep + DTW alignment) in progress; Phase 6D (Conformer translator) next.


> **Update 2026-10-05 (Phase 6D):** Phase 6C & 6D complete. DTW alignment performed across all 3,848 parallel pairs. Conformer accent translator trained and integrated (best val L1: 0.1609 vs 0.2273 baseline), registered in factory as 'conformer'. Ready for full-scale acoustic decoder retrain and streaming adaptation.

