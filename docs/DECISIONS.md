# Decisions

This document records significant architectural decisions, the rationale behind
them, and any rejected alternatives.

---

## DEC-001 — Use ABCs for all model interfaces

**Date**: 2026-09-20  
**Status**: Accepted

### Context

The senior project requires a vocoder for waveform synthesis, but the team also
wants to experiment with TVTSyn or other streaming waveform decoders in later
phases without rewriting pipeline code.

### Decision

Define four abstract base classes (ABCs) in `models/interfaces.py`:
- `SpeechSynthesizer`
- `ContentEncoder`
- `SpeakerEncoder`
- `AccentTranslator`

All pipeline code receives instances typed as the ABC.  Concrete backends are
created only in `models/factory.py`, selected by the `backend` key in the
project configuration.

### Consequences

- Swapping any model backend requires only a YAML config change — no pipeline
  code changes.
- New backends can be added by: (1) creating a new backend file, (2)
  registering it in `factory.py`.
- The interface is enforceable at test time: `isinstance(x, SpeechSynthesizer)`
  verifies compliance without loading any model weights.

### Rejected alternatives

- **Single concrete class with an internal `mode` switch** — rejected because
  it would merge responsibilities and make the class harder to test and extend.
- **Duck typing with no ABC** — rejected because it provides no enforcement
  and makes it possible to silently pass the wrong type to the pipeline.

---

## DEC-002 — Deferred imports in factory.py

**Date**: 2026-09-20  
**Status**: Accepted

### Context

Backend modules will eventually import heavy ML frameworks (PyTorch, etc.).
Importing those at module load time would break any environment that does not
have ML frameworks installed — including CI environments running only audio
tests.

### Decision

All concrete backend class imports in `models/factory.py` are inside functions
(deferred imports), not at the module top level:

```python
def _synthesis_backends():
    from accent_converter.models.backends.synthesis.vocoder import VocoderSynthesizer
    ...
```

This means importing `models.factory` never triggers ML framework imports.
Only calling `build_synthesizer(config)` triggers the backend import.

### Consequences

- Audio and interface tests run without any ML frameworks installed.
- If a backend's import fails (e.g. missing PyTorch), the error is raised at
  the call site with a clear traceback, not at import time.

---

## DEC-003 — Polyphase rational resampling (scipy.signal.resample_poly)

**Date**: 2026-09-20  
**Status**: Accepted  
**Phase**: 2 (Audio Foundation)

### Context

The pipeline must accept 44.1 kHz, 48 kHz, 32 kHz, and other common rates and
convert them to 16 kHz.

### Decision

Use `scipy.signal.resample_poly` with GCD-reduced integer `(up, down)` factors.

### Rationale

- Exact rational resampling avoids aliasing artefacts present in FFT-based
  resampling at non-power-of-two ratios.
- Linear interpolation is fast but introduces phase errors and spectral
  distortion that would degrade content encoding.
- `resample_poly` applies a built-in anti-aliasing FIR filter automatically.

### Consequences

- `scipy` is a runtime dependency (small, well-maintained).
- Resampling is stateless; each call is independent.

---

## DEC-004 — Bounded deque for AudioContextManager

**Date**: 2026-09-20  
**Status**: Accepted  
**Phase**: 2 (Audio Foundation)

### Decision

Use `collections.deque(maxlen=N)` for the sliding-window audio cache.

### Rationale

- O(1) append with automatic oldest-element eviction — exactly the sliding-
  window semantics required.
- No manual index arithmetic or wrap-around logic.
- Deterministic: given the same push sequence, `get()` always returns the same
  result.

---

## DEC-005 — NaiveAccentTranslator as passthrough stub

**Date**: 2026-09-20  
**Status**: Accepted

### Context

The `AccentTranslator` interface is required by the pipeline wiring, but no
real accent translation model is available yet (Phase 6).

### Decision

Provide a `NaiveAccentTranslator` that implements the full interface and
returns the input content representation unchanged.  Its `load()` method is a
no-op (no weights needed) so that end-to-end pipeline tests can be wired
without a real checkpoint.

### Consequences

- The pipeline can be fully assembled and integration-tested without a real
  translation model.
- The naive backend must be replaced before any accent quality evaluation.
- The naive backend is clearly named and documented to prevent accidental
  deployment.

---

## Open questions

| # | Question | Needed by |
|---|----------|-----------|
| Q1 | Which vocoder? HiFi-GAN, UnivNet, or another? Verify checkpoint availability, licence, and streaming support. | Phase 8 |
| Q2 | Is a TVTSyn implementation publicly available with the required streaming properties? | Phase 8 |
| Q3 | Does the HuBERT checkpoint require a specific transformers/fairseq version? | Phase 4 |
| Q4 | Is the ECAPA-TDNN checkpoint from SpeechBrain compatible with the target Python version? | Phase 5 |

| Q5 | What is the target latency budget? Measure before setting targets. | Phase 10 |

---

## DEC-006 — Energy-based VAD instead of webrtcvad

**Date**: 2026-09-21
**Status**: Accepted

### Context

`webrtcvad` and `webrtcvad-wheels` require a C extension compiled against the
MSVC toolchain.  The development machine does not have MSVC installed and
Python 3.14 binary wheels do not exist for these packages.

### Decision

Implement `EnergyVAD` — a pure-numpy, short-time energy VAD with a
configurable hangover counter.

**Parameters**: `threshold_db=-40`, `hangover_frames=8` (640 ms hold-on by default).

### Rationale

- No C extension required → runs on any Python 3.x on any OS.
- Hangover prevents rapid toggling on brief pauses in speech.
- Configurable via YAML — threshold and hangover can be tuned for the accent
  conversion use case without code changes.
- Suitable accuracy for gating model inference; does not need to be a
  reference-quality speech/silence classifier.

### Limitations

- Sensitive to loud stationary non-speech noise (fan, keyboard clicks).
- Should be replaced by a browser WebRTC VAD or an ML-based VAD when
  available.

### Rejected alternatives

- **`webrtcvad`/`webrtcvad-wheels`** — cannot build on this machine (no MSVC).
- **`silero-vad`** — requires PyTorch (not yet a project dependency; deferred to Phase 4+).

---

## DEC-007 — noisereduce for noise suppression

**Date**: 2026-09-21
**Status**: Accepted

### Decision

Use `noisereduce.reduce_noise` (spectral gating) as the `NoiseReduceNoiseSuppressor` backend.

### Rationale

- Pure Python + numpy + scipy — installs cleanly on Python 3.14 Windows.
- Supports both stationary and non-stationary noise models.
- Requires no model checkpoint — works out of the box.
- Spectral gating is well-understood and sufficient for the Phase 3 prototype.

### Limitations

- Not frame-synchronous: processes the full input buffer per call.
  Chunk-by-chunk processing (80 ms) may produce boundary artefacts.
  Recommendation: accumulate ≥ 500 ms before applying suppression, or apply
  post-VAD on complete speech segments.
- No persistent noise model across calls (each call re-estimates the noise).

### Rejected alternatives

- **RNNoise** — requires building a C library; no Python 3.14 wheels.
- **DeepFilterNet** — requires PyTorch; deferred to Phase 4+.

---

## DEC-008 — PassthroughAEC with far-end reference in interface

**Date**: 2026-09-21
**Status**: Accepted

### Decision

The `AcousticEchoCanceller` interface includes a `far_end` parameter in
`process()` even though the only current backend (`PassthroughAEC`) ignores it.

### Rationale

- Locks in the correct API now, before browser integration.
- The real AEC (browser-side WebRTC AEC3) will be transparent to the pipeline:
  audio arrives at the Python backend already echo-cancelled.
- `PassthroughAEC` warns when `far_end` is supplied to catch accidental misuse.
- Future server-side AEC (e.g. `SpeexAEC`) can be plugged in by adding one
  file and one registry entry — zero pipeline changes.

---

## DEC-009 — Preprocessing is independent of the ML model layer

**Date**: 2026-09-21
**Status**: Accepted

### Decision

All preprocessing components (`VoiceActivityDetector`, `NoiseSuppressor`,
`AcousticEchoCanceller`) live in `audio/preprocessing/` — not in
`models/`.  They have their own `interfaces.py` and `factory.py`.

### Rationale

- Preprocessing runs on raw audio before any ML inference.
- Preprocessing has no `load(checkpoint_path)` method: it does not load ML
  weights (energy VAD and spectral gating are algorithmic, not learned).
- Keeping the two layers separate makes it easier to profile and optimise
  each independently.

---

## DEC-010 — Two-tier content encoder strategy and metadata properties

**Date**: 2026-09-21
**Status**: Accepted
**Phase**: 4 (Content Encoder Research and Validation)

### Context

The whiteboard specification calls for HuBERT, but HuBERT uses bidirectional
self-attention over the entire utterance and is not causal. Streaming cannot be
achieved simply by feeding chunks to standard HuBERT. A streaming-compatible
causal encoder architecture is required for the real-time target.

### Decision

1. Adopt a two-tier content encoder strategy:
   - **Offline Reference (Phases 4–6)**: HuBERT Base (`facebook/hubert-base-ls960`)
     via HuggingFace `transformers` for offline validation, feature verification,
     and accent translator prototyping.
   - **Streaming Production (Phase 7)**: TVTSyn-style causal encoder (causal 1-D CNN
     + causal MHSA with ring KV-cache + VQ bottleneck), implemented from scratch
     since no public checkpoint exists.
2. Extend `ContentEncoder` interface with three abstract properties:
   - `frame_rate_hz: int` (50 Hz for HuBERT)
   - `output_dim: int` (768 for HuBERT Base)
   - `lookahead_ms: int` (-1 sentinel for non-causal offline models)
3. Do not couple the streaming pipeline loop to HuBERT directly.

### Consequences

- Offline pipeline validation can proceed without waiting for custom causal
  encoder training.
- All 205 unit tests verify the updated interface contract.

---

## DEC-011 — Silero VAD as primary neural VAD backend

**Date**: 2026-09-24
**Status**: Accepted
**Phase**: Preprocessing (Upgrade)

### Context

Originally, `EnergyVAD` was chosen in DEC-006 because `webrtcvad` required C compilation (unavailable on Python 3.14 Windows) and PyTorch had not yet been introduced into the environment.

With Phases 4 & 5 complete, PyTorch is now an established runtime dependency. `EnergyVAD` operates strictly on RMS amplitude and cannot distinguish speech from loud acoustic non-speech noises (keyboard clatter, fan noise, microphone bumps, music).

### Decision

1. Implement `SileroVAD` (`src/accent_converter/audio/preprocessing/vad/silero.py`) as the default Voice Activity Detector backend.
2. Package: `silero-vad>=6.0` (bundled local weights, zero network download at runtime).
3. Streaming window adapter: Buffer incoming audio chunks (canonically 80 ms = 1,280 samples) and feed complete 512-sample windows to the model; maintain hangover hold-on counter.
4. Retain `EnergyVAD` as an algorithmic fallback.
5. Set `preprocessing.vad.backend: "silero"` in `configs/default_config.yaml`.

### Consequences

- Highly robust speech/non-speech gating resistant to background noise and pure tones.
- No network downloads required during test execution or deployment.
- Seamless drop-in replacement via the `VoiceActivityDetector` ABC.