# Model Selection

> This document records final and pending selection decisions for all model
> components in the accent-converter pipeline.
> It is updated after each research phase.

---

## Selection status

| Component | Phase | Selection | Status |
|---|---|---|---|
| Content Encoder (offline ref) | 4B | HuBERT Base (`facebook/hubert-base-ls960`) | ✅ **Implemented & Verified** |
| Content Encoder (streaming) | 7 | TVTSyn-style causal CNN + causal MHSA | **Architecture defined — needs implementation** |
| Speaker Encoder | 5 | ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`) | ✅ **Implemented & Verified** |
| Accent Translator (stub) | 6A | NaiveAccentTranslator (passthrough) | ✅ **Implemented** |
| Accent Translator (real) | 6B | TBD (seq2seq / non-parallel method) | Phase 6B research |
| Waveform Synthesizer (primary) | 8 | Vocoder (HiFi-GAN + Acoustic Decoder) | Phase 8 target (stub exists) |
| Waveform Synthesizer (alt) | 8 | TVTSyn decoder | Stub exists |
| VAD | 3 | SileroVAD (neural default) / EnergyVAD (fallback) | ✅ **Implemented & Verified** |
| Noise Suppression | 3 | NoiseReduceNoiseSuppressor (spectral gating) | ✅ **Implemented** |
| AEC | 3/11 | PassthroughAEC (browser WebRTC in Phase 11) | ✅ **Wired** |

---

## Content Encoder — detailed selection

### Phase 4 (offline reference): HuBERT Base

**Selected**: `facebook/hubert-base-ls960` via HuggingFace `transformers`.

**Reason**: Only available production checkpoint for a content encoder that
matches the project's architecture (SSL, speaker-disentangled features).
Apache 2.0 license; well-maintained; 90M parameters; ~2–4 GB VRAM.

**Role**: Offline reference encoder for Phase 4–6 pipeline validation.
Must NOT be used as the live streaming encoder.

**Explicit non-streaming justification**: HuBERT uses full bidirectional
self-attention.  Its representation at frame $t$ depends on all future frames.
Producing output for frame $t$ requires the entire utterance.  Calling it on
chunks does not make it streaming — the representations will be inconsistent
between calls.

**Lookahead**: Unbounded (non-causal). `lookahead_ms = -1` (sentinel value
meaning "not causal").

---

### Phase 7 (streaming production): TVTSyn-style causal encoder

**Selected architecture**: Causal 1-D CNN front-end + 8 × causal MHSA blocks
with ring KV cache (2-second window) + VQ bottleneck (8-dim codebook).

**Reason**: The only architecture in the literature that satisfies all of:
- Genuinely causal (bounded past + small fixed future ≤80 ms).
- Sub-100 ms GPU latency demonstrated.
- Designed for voice conversion (same task family).
- VQ bottleneck provides speaker disentanglement.

**Availability gap**: No public code or checkpoint exists.  This component
must be built from scratch.

**Not a blocking issue for Phases 4–6** — the offline HuBERT encoder serves
as a fully functional reference during those phases.

---

## Speaker Encoder — pending research

- Candidate: ECAPA-TDNN (SpeechBrain).
- Status: Stub exists.  Dedicated research phase (Phase 5) pending.
- Open question: SpeechBrain + Python 3.14 compatibility on Windows.

---

## Accent Translator — pending research

- Current: NaiveAccentTranslator (passthrough, for pipeline wiring only).
- Candidate approaches: seq2seq on phoneme tokens, non-parallel CTC,
  PPG-based translator.
- Status: Research deferred until offline content encoder is functional
  (Phase 6 dependency).

---

## Waveform Synthesizer — pending research

- Primary: vocoder (senior project requirement).
  - Candidate: HiFi-GAN, UnivNet, BigVGAN.
  - Open question: which supports streaming / chunk-by-chunk inference.
- Alternative: TVTSyn decoder (same paper as the causal content encoder).
- Status: Research deferred until Phase 8.

---

## Selection criteria (reference)

The following criteria are applied to all model selections:

| Criterion | Importance |
|---|---|
| Genuine streaming support (bounded causal) | Critical for final system |
| Available public checkpoint | Required for Phase 4–6 |
| License permits research + educational use | Required |
| Dependencies compatible with Python 3.14 / Windows | Required |
| VRAM within reasonable budget (≤8 GB) | High |
| Pre-trained on / generalisable to Indian-accented English | High |
| Speaker-disentangled output | High |
| Published performance benchmarks | Medium |
| Code quality / maintainability | Medium |

---

## Open questions

| # | Question | Blocks |
|---|---|---|
| Q1 | Does HuBERT Base represent Indian-accented English adequately without fine-tuning? | Phase 4 evaluation |
| Q2 | What VRAM is available on the target deployment GPU? | All Phase 4+ |
| Q3 | Is SpeechBrain (ECAPA-TDNN) compatible with Python 3.14 on Windows? | Resolved ✅ (SpeechBrain 1.1.1 working, 25 tests pass) |
| Q4 | Which vocoder supports chunk-by-chunk streaming without full-utterance context? | Phase 8 |
| Q5 | Can a causal CNN + MHSA encoder be trained to match HuBERT feature quality? | Phase 7 |
| Q6 | What Indian-accented English dataset will be used for training? | Phase 6B / 7+ |
