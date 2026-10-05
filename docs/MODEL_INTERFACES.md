# Model Interfaces

> **Status**: Interfaces defined and verified.
> Concrete implementations complete: `HubertContentEncoder` (Phase 4B) and `EcapaSpeakerEncoder` (Phase 5).
> Phase 6A NaiveAccentTranslator passthrough stub implemented; Phase 6B translator and Phase 8 synthesizer pending.

---

## Overview

Every model component in the pipeline is accessed through an **abstract base class (ABC)**.
The pipeline never imports a concrete backend class directly.
The factory reads configuration and returns the appropriate instance.

```
Configuration (YAML)
      │
      ▼
   factory.py           ← build_synthesizer(config["synthesis"])
      │                    build_content_encoder(config["models"]["content_encoder"])
      │                    build_speaker_encoder(config["models"]["speaker_encoder"])
      │                    build_accent_translator(config["models"]["accent_translator"])
      ▼
  Interface (ABC)       ← what pipeline code sees
      │
      ▼
  Backend (stub/impl)   ← concrete class selected by factory
```

---

## Interface Definitions

### `SpeechSynthesizer`

**File**: [`src/accent_converter/models/interfaces.py`](../src/accent_converter/models/interfaces.py)

The primary waveform synthesis interface. The pipeline depends **only** on this type.

```python
class SpeechSynthesizer(ABC):
    def load(self, checkpoint_path: str) -> None: ...
    def synthesize(
        self,
        converted_rep: np.ndarray,      # output of AccentTranslator
        speaker_embedding: np.ndarray,  # output of SpeakerEncoder
    ) -> np.ndarray: ...                # 1-D float32, 16 kHz
    def reset(self) -> None: ...
```

**Registered backends**:

| Backend name | Class | Status |
|---|---|---|
| `vocoder` | `VocoderSynthesizer` | Stub — Phase 8 |
| `tvtsyn` | `TVTSynSynthesizer` | Stub — Phase 8 |

**Config**:
```yaml
synthesis:
  backend: vocoder    # or: tvtsyn
  checkpoint: null
```

---

### `ContentEncoder`

```python
class ContentEncoder(ABC):
    def load(self, checkpoint_path: str) -> None: ...
    def encode(self, audio: np.ndarray) -> np.ndarray: ...
    def reset(self) -> None: ...
```

**Input**: 1-D float32 audio at 16 kHz  
**Output**: content representation (shape/frame-rate is backend-specific)

**Registered backends**:

| Backend name | Class | Notes |
|---|---|---|
| `hubert` | `HubertContentEncoder` | Implemented (Phase 4B) — offline reference; non-causal (50 Hz, 768-dim) |

> ⚠️ Standard HuBERT is **not causal**. Do not use it as a streaming encoder
> without explicit causal adaptation.

**Config**:
```yaml
models:
  content_encoder:
    backend: hubert
    checkpoint: null
```

---

### `SpeakerEncoder`

```python
class SpeakerEncoder(ABC):
    def load(self, checkpoint_path: str) -> None: ...
    def encode(self, reference_audio: np.ndarray) -> np.ndarray: ...
    def reset(self) -> None: ...
```

**Input**: 1-D float32 enrollment/reference audio at 16 kHz  
**Output**: fixed-dimension speaker embedding vector (192-dim float32)

Compute the embedding once per enrollment; cache and reuse during the call.

**Registered backends**:

| Backend name | Class | Notes |
|---|---|---|
| `ecapa` | `EcapaSpeakerEncoder` | Implemented (Phase 5) — SpeechBrain ECAPA-TDNN (192-dim) |

**Config**:
```yaml
models:
  speaker_encoder:
    backend: ecapa
    checkpoint: null
```

---

### `AccentTranslator`

```python
class AccentTranslator(ABC):
    def load(self, checkpoint_path: str) -> None: ...
    def translate(
        self,
        content_rep: np.ndarray,     # output of ContentEncoder
        target_accent: str,          # "us" | "uk" | "neutral"
    ) -> np.ndarray: ...             # accent-converted representation
    def reset(self) -> None: ...
```

**Canonical data flow**:

```text
ContentEncoder.encode(audio)         → content_rep
AccentTranslator.translate(
    content_rep, target_accent)      → target_rep
SpeechSynthesizer.synthesize(
    target_rep, speaker_embedding)   → waveform
```

**Speaker embedding belongs at the Synthesizer, not the Translator.**
The Translator is responsible only for accent/pronunciation transformation.
Speaker identity conditioning is applied by the Synthesizer.

**Purpose**: accent/pronunciation transformation while preserving linguistic content  
**Must not**: perform ASR, TTS, speaker conversion, or access the speaker embedding

**Registered backends**:

| Backend name | Class | Notes |
|---|---|---|
| `naive` | `NaiveAccentTranslator` | Passthrough — returns content_rep unchanged |

**Config**:
```yaml
models:
  accent_translator:
    backend: naive
    checkpoint: null
    target_accent: us    # options: us | uk | neutral
```

---

## Adding a New Backend

1. Create `src/accent_converter/models/backends/<role>/<name>.py`
2. Subclass the appropriate ABC from `models.interfaces`
3. Implement all abstract methods (`load`, the role method, `reset`)
4. Add a `BACKEND_NAME: str` class attribute
5. Register it in the corresponding `_*_backends()` dict in `models/factory.py`
6. Update `configs/default_config.yaml` with the new backend name
7. Write tests for the new backend's interface compliance

**The pipeline requires zero changes.**

---

## Interface Contracts

All four interfaces enforce the following contracts:

| Requirement | Mechanism |
|---|---|
| Cannot instantiate ABC directly | `ABC` + `@abstractmethod` → `TypeError` |
| `load()` must be called first | Implementer enforces; stubs raise `NotImplementedError` |
| `reset()` is always safe | Must not raise, even before `load()` |
| Output is `np.ndarray` | Typed in the method signature |
| No ML framework in pipeline code | Concrete imports only in backend files |

---

## Test Coverage

Interface tests: [`tests/unit/test_model_interfaces.py`](../tests/unit/test_model_interfaces.py)

| Test class | What is tested |
|---|---|
| `TestABCEnforcement` | ABCs cannot be instantiated without full implementation |
| `TestPipelineDecoupling` | Pipeline accepts any `SpeechSynthesizer` without code changes |
| `TestFactory` | Factory returns correct type; rejects unknown/missing backend |
| `TestConcreteStubs` | Stubs satisfy `isinstance`, raise correctly, passthrough works |
| `TestResetSafety` | `reset()` safe before `load()`, safe to call multiple times |