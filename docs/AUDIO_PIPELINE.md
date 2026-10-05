# Audio Pipeline — Phase 2: Audio Foundation

> **Status**: Implemented and tested (Phase 2 Audio Foundation & Phase 3 Audio Preprocessing).
> Includes SileroVAD (default neural VAD), EnergyVAD (fallback), NoiseReduceNoiseSuppressor, and PassthroughAEC.

---

## 1. Overview

The audio foundation converts raw microphone input into a canonical internal
representation and feeds it to the downstream model pipeline in fixed-size
chunks with a bounded look-back context.

```
Raw audio (any SR, any dtype, mono or stereo)
        │
        ▼
  ┌─────────────┐
  │  Resampler  │  resample_poly to 16 000 Hz
  └─────────────┘
        │
        ▼
  ┌──────────────┐
  │  Normalizer  │  to_mono → to_float32 → peak_normalize
  └──────────────┘
        │
        ▼
  ┌─────────┐
  │ Chunker │  80 ms frames = 1 280 samples @ 16 kHz
  └─────────┘
        │
        ▼
  ┌──────────────────────┐
  │ AudioContextManager  │  sliding window, bounded look-back
  └──────────────────────┘
        │
        ▼
  Downstream model (Phase 4+)
```

---

## 2. Canonical Audio Format

| Property    | Value           |
|-------------|-----------------|
| Sample rate | **16 000 Hz**   |
| Channels    | **mono** (1-D)  |
| dtype       | **float32**     |
| Amplitude   | **≈ [−1, 1]**   |

All audio entering the model pipeline **must** be in this format. The resampler
and normalizer enforce it.

---

## 3. Module Responsibilities

### `audio.resampler` — Sample-rate conversion

**File**: [`src/accent_converter/audio/resampler.py`](../src/accent_converter/audio/resampler.py)

- Uses `scipy.signal.resample_poly` (polyphase rational resampling).
- GCD-reduces `target_sr / orig_sr` to minimize filter length.
- Passthrough (copy) when `orig_sr == target_sr == 16 000`.
- Accepts any positive integer sample rate.
- **Tested rates**: 8 kHz, 16 kHz, 22.05 kHz, 32 kHz, 44.1 kHz, 48 kHz, 96 kHz.
- **Stateless** — each call is independent.

```python
from accent_converter.audio.resampler import resample

audio_16k = resample(audio_44k, orig_sr=44_100)   # → float32 1-D @ 16 kHz
```

---

### `audio.normalizer` — Canonical normalisation

**File**: [`src/accent_converter/audio/normalizer.py`](../src/accent_converter/audio/normalizer.py)

| Function           | Input               | Output                    |
|--------------------|---------------------|---------------------------|
| `to_mono(audio)`   | any 1-D or 2-D arr  | 1-D, same dtype           |
| `to_float32(audio)`| any numeric dtype   | float32, scaled to [−1,1] |
| `peak_normalize()` | float32 1-D         | float32 1-D, peak = 1.0   |
| `normalize(audio)` | any shape/dtype     | canonical float32 1-D     |

Integer-to-float scaling table:

| Input dtype | Scale factor    |
|-------------|-----------------|
| int16       | ÷ 32 768        |
| int32       | ÷ 2 147 483 648 |
| uint8       | (x − 128) ÷ 128 |
| float16/64  | cast only       |

Silent frames (peak < 1×10⁻⁸) are **not rescaled** to avoid amplifying silence.

---

### `audio.chunker` — Fixed-size framing

**File**: [`src/accent_converter/audio/chunker.py`](../src/accent_converter/audio/chunker.py)

```
At 16 000 Hz:   80 ms × 16 000 = 1 280 samples per chunk
```

Constants exported:

```python
CHUNK_MS      = 80       # nominal duration in milliseconds
CHUNK_SAMPLES = 1_280    # exact sample count at 16 kHz
```

**API**:

```python
chunk_audio(audio, chunk_size_ms=80, sample_rate=16_000, tail="drop")
# Returns list[np.ndarray]

frame_generator(audio, chunk_size=1_280, tail="drop")
# Generator variant — memory-efficient for long streams
```

**`tail` parameter**:

| Value    | Behaviour                                        |
|----------|--------------------------------------------------|
| `"drop"` | Discard leftover samples (default)               |
| `"keep"` | Return partial chunk at its actual length        |
| `"pad"`  | Zero-pad partial chunk to `chunk_size` samples   |

> **Note**: capture chunk size and model frame rate are not necessarily the
> same. Do not redesign model timing because the browser delivers a different
> chunk size.

---

### `audio.context` — Bounded streaming context

**File**: [`src/accent_converter/audio/context.py`](../src/accent_converter/audio/context.py)

```python
from accent_converter.audio.context import AudioContextManager

ctx = AudioContextManager(max_frames=8, chunk_samples=1_280)

for chunk in stream:
    ctx.push(chunk)        # oldest frame auto-evicted when full
    window = ctx.get()     # chronological concat of all buffered frames
    run_model(window)

ctx.reset()                # start of a new utterance/call
```

**Guarantees**:

| Property      | Detail                                                         |
|---------------|----------------------------------------------------------------|
| Bounded memory | Buffer never exceeds `max_frames × chunk_samples` samples     |
| Deterministic  | Same push sequence → same `get()` output, always              |
| Reset          | `reset()` fully clears state, no residue between sessions      |
| Copy-on-push   | External mutation of pushed array does not corrupt the cache   |

**Default parameters**:

| Parameter      | Default | Meaning                          |
|----------------|---------|----------------------------------|
| `max_frames`   | 8       | 8 × 80 ms = 640 ms look-back     |
| `chunk_samples`| 1 280   | Expected samples per frame        |

---

## 4. Usage Example

```python
import numpy as np
from accent_converter.audio import (
    resample,
    normalize,
    chunk_audio,
    AudioContextManager,
    CHUNK_SAMPLES,
)

# Simulate raw 48 kHz int16 stereo microphone input
raw = np.random.randint(-32768, 32767, size=(2, 48_000), dtype=np.int16)

# Step 1: Mono + float32 conversion
mono_f32 = normalize(raw, peak_norm=False)       # (48000,) float32

# Step 2: Resample to 16 kHz
audio_16k = resample(mono_f32, orig_sr=48_000)   # (16000,) float32 (1 s)

# Step 3: Peak normalise
from accent_converter.audio.normalizer import peak_normalize
audio_norm = peak_normalize(audio_16k)

# Step 4: Chunk into 80 ms frames
chunks = chunk_audio(audio_norm)   # list of (1280,) arrays

# Step 5: Feed with look-back context
ctx = AudioContextManager(max_frames=8)
for chunk in chunks:
    ctx.push(chunk)
    window = ctx.get()    # up to 640 ms of recent audio
    # → feed to model ...

ctx.reset()   # new call / utterance
```

---

## 5. Design Decisions

### Why `resample_poly`?

`scipy.signal.resample_poly` uses polyphase FIR filtering with exact rational
up/down factors. This avoids:
- aliasing from FFT-based resampling at non-power-of-two ratios,
- phase errors from linear interpolation,
- spectral leakage.

The GCD reduction (e.g. 48 000 / 16 000 = 3/1) keeps the filter short.

### Why no single "AudioProcessor" class?

Resampling, normalisation, chunking, and context management have **different
lifetimes and statefulness**:

- Resampling and normalisation are **stateless** per-call operations.
- Chunking is **stateless** frame extraction.
- Context management is **stateful** and must be reset between sessions.

Combining them into one class would violate single-responsibility and make
unit testing harder.

### Why `deque(maxlen=N)` for the context buffer?

Python's `collections.deque` with `maxlen` provides O(1) append with
automatic eviction of the oldest element — exactly the sliding-window
semantics needed, without manual index arithmetic.

---

## 6. Test Coverage

Run with:

```bash
python -m pytest tests/unit/test_audio_normalizer.py tests/unit/test_audio_resampler.py tests/unit/test_audio_chunker.py tests/unit/test_audio_context.py -v
```

| Test file                    | What is tested                                      |
|------------------------------|-----------------------------------------------------|
| `test_audio_normalizer.py`   | to_mono, to_float32, peak_normalize, normalize      |
| `test_audio_resampler.py`    | 44.1/48/32/8 kHz → 16 kHz, passthrough, errors      |
| `test_audio_chunker.py`      | 80 ms chunk size, tail modes, sequential chunks      |
| `test_audio_context.py`      | cache limit, eviction, reset, determinism, validation|

---

## 7. Status of Downstream Components
- **Audio Preprocessing (Phase 3)**: ✅ **Implemented** (`SileroVAD` neural default, `EnergyVAD` fallback, `NoiseReduceNoiseSuppressor`, `PassthroughAEC`).
- **Content Encoder (Phase 4B)**: ✅ **Implemented** (`HubertContentEncoder` offline reference, (T, 768)).
- **Speaker Encoder (Phase 5)**: ✅ **Implemented** (`EcapaSpeakerEncoder`, (192,)).
- **Accent Translator**: Phase 6A passthrough stub implemented; Phase 6B learned model pending.
- **Speech Synthesizer**: Phase 8 target (stub exists).
- **Real-Time Streaming / Transport**: Phases 7, 9, 10, 11 pending.

---

## 8. Related Documents

- [`docs/BASELINE.md`](BASELINE.md) — repository baseline status
- [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — overall system architecture
- [`ACCENT_CONVERTER_AGENT_INITIAL_SETUP.md`](../ACCENT_CONVERTER_AGENT_INITIAL_SETUP.md) — canonical audio format specification