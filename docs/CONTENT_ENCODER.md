# Content Encoder — Research and Selection

> **Phase 4 status**: Phase 4B (HuBERT Offline Reference Encoder) IMPLEMENTED.
> Empirical measurements recorded below. Streaming suitability evaluated.

---

## Research objective

The accent-converter pipeline requires a content encoder that extracts
**linguistic/content representations intended for use in speaker/accent disentanglement**
from raw audio. The representation must be suitable for accent translation without
carrying speaker-identity information into the translator.
The actual degree of speaker information retained should be evaluated experimentally.

Because the final system is **streaming** (no complete utterance available),
the candidate must be evaluated for **genuine streaming suitability**, not
merely the ability to process audio in chunks.  A model that requires the
full sequence for its attention mechanism is **not** streaming regardless of
whether it is called in a loop.

---

## Candidate 1 — HuBERT (Hidden-Unit BERT)

### Purpose
Self-supervised learning of speech representations via offline BERT-like
masked prediction over cluster-assigned pseudo-labels.  Used widely as an
offline feature extractor for downstream tasks (ASR, voice conversion, etc.).

### Technical facts

| Property | Value | Source |
|---|---|---|
| Input sample rate | 16 000 Hz | Paper / HuggingFace model card |
| Input format | 1-D float32, normalised | `AutoProcessor` |
| Output representation | Dense float32 tensor, shape `(T_frames, D)` | `last_hidden_state` |
| Output dimensionality (base) | 768 | 12 transformer layers |
| Output dimensionality (large) | 1024 | 24 transformer layers |
| Frame rate | 50 Hz (one frame per 20 ms) | Conv feature encoder stride |
| Conv front-end receptive field | ~25 ms (400 samples) | 7 strided convolutions |
| **Attention type** | **Bidirectional full-context self-attention** | Architecture |
| **Lookahead** | **Unbounded — attends to entire sequence** | Architecture |
| **Causal** | **No** | Architecture |

### Streaming suitability

**Standard HuBERT is NOT suitable for streaming without architectural
modification.**

The Transformer encoder uses full bidirectional self-attention.  Each frame
attends to all other frames including all future frames.  This means:

- The final representation of frame $t$ depends on frames $t+1, t+2, \ldots, T$.
- You cannot produce a committed output for frame $t$ until the entire
  utterance has been encoded.
- Calling `model(chunk)` on 80 ms chunks produces representations that change
  when subsequent chunks arrive.

This is not "streaming" in the sense required by this project.

### Streaming variants (SHuBERT / LLSA)

Research has produced streaming-capable modifications:
- Replace bidirectional attention with **causal masking + bounded look-ahead**.
- Use **Low-Latency Streaming Attention (LLSA)** modules.
- Fine-tune with causal constraint (LoRA or full fine-tune).

These variants are:
- **Not officially released** as production checkpoints at the time of writing.
- Require additional training on top of the standard HuBERT checkpoint.
- Not trivially adaptable without custom PyTorch code.

### Role in this project

HuBERT Base is the appropriate **offline reference / teacher** model for Phase 4
development:

1. Use it to verify the content encoder interface works end-to-end (offline).
2. Use its representations as targets for a causal student encoder (Phase 7).
3. Never use it as the live streaming encoder.

### Available implementation

| Item | Value |
|---|---|
| HuggingFace checkpoint | `facebook/hubert-base-ls960` |
| Parameters | ~90 M |
| Checkpoint size (disk) | ~360 MB |
| GPU memory (inference, fp32) | ~2–4 GB VRAM |
| CPU inference | Possible but slow (multiple seconds per utterance) |
| License | Apache 2.0 |
| Python dependency | `transformers[torch]` + `torch` |
| Python 3.14 compatible | Yes (PyTorch 2.x, transformers 4.x/5.x) |

### Integration complexity

- **Low** for offline use: load via explicit classes `Wav2Vec2FeatureExtractor` and `HubertModel`, call `model(input_values).last_hidden_state`.
- **High** for streaming: requires architectural modification + retraining.

### Empirical Benchmarks (Phase 4B Measured)

The following measurements were obtained empirically using `scripts/benchmark_hubert.py` on the real `facebook/hubert-base-ls960` checkpoint:

- **Host Environment**: Windows (Python 3.14.5, `torch==2.14.0+cu126`, `transformers==5.17.0`)
- **GPU Hardware**: NVIDIA GeForce RTX 3050 6GB Laptop GPU (6.00 GB total VRAM)
- **Model Parameters**: 94.68 M
- **Output Dimensionality**: 768 (`last_hidden_state`)
- **Nominal Frame Rate**: 50 Hz (`lookahead_ms = -1`, non-causal offline reference)
- **Model Load Time**:
  - CPU: **7.394 s**
  - CUDA: **2.900 s**
- **Peak CUDA Memory (VRAM)**: **450.6 MB**

#### Latency, Frame Counts & Real-Time Factor (RTF)

| Duration | Measured Frames | Frame Rate (obs) | CPU Latency (ms) | CPU RTF | CUDA Latency (ms) | CUDA RTF |
|:---|:---|:---|:---|:---|:---|:---|
| **1.0 s** | 49 | 49.00 Hz | 60.01 ms | 0.0600 | 11.10 ms | 0.0111 |
| **3.0 s** | 149 | 49.67 Hz | 144.49 ms | 0.0482 | 18.78 ms | 0.0063 |
| **5.0 s** | 249 | 49.80 Hz | 225.17 ms | 0.0450 | 25.45 ms | 0.0051 |

> [!NOTE]
> **Actual Frame Counts**: HuBERT's 7-layer convolutional feature encoder uses kernel sizes (10, 3, 3, 3, 3, 2, 2) and strides (5, 2, 2, 2, 2, 2, 2), yielding receptive field 400 samples (25 ms) and cumulative stride 320 samples (20 ms). For unpadded raw waveforms of $N$ samples, $T = \lfloor (N - 400)/320 \rfloor + 1$. Thus, 16 000 samples (1.0 s) produces exactly **49 frames**, 48 000 samples (3.0 s) produces **149 frames**, and 80 000 samples (5.0 s) produces **249 frames**. Frame counts are measured directly from the model output rather than assumed.

### Known limitations

- Not disentangled by design: representations retain some speaker information.
- Bidirectional context; not causal.
- Pre-trained on English LibriSpeech; may not generalise well to Indian-
  accented English without fine-tuning (evaluate empirically).

---

## Candidate 2 — WavLM

### Purpose
Extension of HuBERT with gated relative positional encoding and denoising
pre-training objective.  State-of-the-art on SUPERB benchmark as of original
publication.

### Technical facts

| Property | Value |
|---|---|
| Input sample rate | 16 000 Hz |
| Input format | 1-D float32 |
| Output representation | Dense float32 `(T_frames, D)` |
| Frame rate | 50 Hz (20 ms/frame, same stride as HuBERT) |
| Output dimensionality (base) | 768 |
| Output dimensionality (large) | 1024 |
| Conv front-end receptive field | ~25 ms |
| **Attention type** | **Gated relative-position bidirectional self-attention** |
| **Lookahead** | **Unbounded** |
| **Causal** | **No** |

### Streaming suitability

**Standard WavLM is NOT suitable for streaming for the same reason as HuBERT.**

The gated relative-position encoding does not make it causal; it still uses
full bidirectional self-attention.

### Streaming variant: WavSLM

Research on **WavSLM** proposes a streaming adaptation:
- Replace standard convolutions with causal convolutions.
- Replace full-context gated relative attention with
  **sliding-window gated relative chunked attention**.
- Achieved theoretical latency of ~80 ms in the studied configuration.
- This is a **research paper** — no production checkpoint released.

### Role in this project

- Same offline teacher role as HuBERT.
- WavLM Large typically outperforms HuBERT Large on downstream tasks.
- May be preferred over HuBERT Base if GPU budget permits (315M parameters,
  ~1.2 GB checkpoint, 4–8 GB VRAM).
- **Not the streaming encoder.**

### Available implementation

| Item | Value |
|---|---|
| HuggingFace checkpoint | `microsoft/wavlm-base`, `microsoft/wavlm-large` |
| License | MIT |
| Python dependency | `transformers[torch]` + `torch` |
| Python 3.14 compatible | Yes |

### Integration complexity

- Same as HuBERT for offline use.
- Streaming requires WavSLM-style retraining — higher complexity than HuBERT
  causal adaptation.

### Advantage over HuBERT

- Denoising objective helps WavLM learn more noise-robust representations.
- Relevant for Indian-accented speech in noisy call environments.
- Gated relative positional bias improves sequence-position handling.

---

## Candidate 3 — TVTSyn-style causal content encoder

### Purpose
The TVTSyn architecture (Content-Synchronous Time-Varying Timbre, 2026)
was specifically designed for **streaming voice conversion and anonymization**
with sub-100 ms GPU latency.  It includes a built-for-streaming causal content
encoder.

### Technical facts

| Property | Value | Evidence |
|---|---|---|
| Input sample rate | 16 000 Hz | Paper |
| Input format | Raw waveform | Paper |
| Encoder type | Causal 1-D CNN + 8 causal MHSA blocks | Paper |
| **Lookahead** | ~80 ms (small fixed future window for coarticulation) | Paper |
| **Causal** | **Yes** (bounded lookahead only) | Architecture |
| Context window | 2-second rolling KV cache | Paper |
| Output | Quantised content tokens (VQ bottleneck, 8-dim) | Paper |
| End-to-end GPU latency | < 80 ms | Paper |
| VQ bottleneck | 8-dimensional learnable codebook | Paper — speaker disentanglement |
| **Streaming** | **Genuinely streaming** | Architecture |

### Why it is genuinely streaming

The causal attention mask in the MHSA blocks restricts attention to:
- **All past frames** (via ring KV cache, bounded to 2 s).
- **A small fixed future window** (~80 ms) for coarticulation.

This means:
- Output at frame $t$ depends only on frames in $[t-2s, t+80ms]$.
- Bounded memory.
- Bounded latency.
- Can process live audio without waiting for utterance end.

This satisfies the definition of streaming used in this project.

### The PHONOS system (built on TVTSyn)

PHONOS (PHOnetic Neutralization for Online Streaming Applications, Interspeech
2026) builds directly on TVTSyn for **foreign accent conversion**:

- Uses the TVTSyn causal content encoder as the representation backbone.
- Trains a **causal accent translator** on top with ≤40 ms look-ahead.
- End-to-end GPU latency: 241 ms.
- Uses silence-aware DTW alignment and joint cross-entropy + CTC losses.
- Authors: Waris Quamer, Mu-Ruei Tseng, Ghady Nasrallah, Ricardo Gutierrez-Osuna.

### Availability

| Item | Status |
|---|---|
| TVTSyn paper | arXiv, February 2026 |
| TVTSyn code | **No public repository** |
| TVTSyn checkpoint | **Not released** |
| PHONOS paper | Interspeech 2026 |
| PHONOS code | **No public repository** |
| PHONOS checkpoint | **Not released** |
| Paper license | CC BY 4.0 (document only) |

### Integration risk

**HIGH** — Neither TVTSyn nor PHONOS has a public implementation.

Building a TVTSyn-equivalent content encoder requires:
1. Implementing a causal 1-D CNN front-end from scratch.
2. Implementing 8 causal MHSA blocks with ring KV cache.
3. Implementing a VQ bottleneck.
4. Training from scratch on accent-conversion data (no available checkpoint).
5. Validating that the representation is speaker-disentangled.

This is a **substantial research engineering task**, not an integration task.

### Role in this project

TVTSyn's causal encoder is the **target architecture for the production
streaming encoder** (Phase 7+).  However:

- It cannot be integrated before an offline reference pipeline exists (Phase 6).
- A simpler causal encoder (e.g., causal HuBERT adapter or a causal CNN) may
  serve as an interim streaming encoder while the full TVTSyn variant is built.

---

## Candidate 4 — PHONOS-style causal content encoding with phoneme tokens

### Purpose
Rather than using dense SSL features, PHONOS-style encoding produces
**discrete phoneme-like tokens** via a VQ bottleneck on top of a causal
encoder.  This makes the representation more linguistically interpretable and
potentially more suitable for accent translation.

### Technical facts

| Property | Value |
|---|---|
| Representation type | Discrete phoneme tokens (VQ) |
| Token rate | ~50 tokens/second (frame-synchronous) |
| Causal | **Yes** (same causal encoder as TVTSyn) |
| Lookahead | ≤40 ms in PHONOS accent translator |
| Advantage | More interpretable; smaller VQ vocabulary |
| Disadvantage | Requires training a VQ codebook |
| Streaming | **Genuinely streaming** |

### Relation to TVTSyn

This is essentially the TVTSyn content encoder output **post-quantisation**.
The encoder is the same; the quantisation layer produces tokens instead of
continuous dense embeddings.

### Availability

Same as TVTSyn — no public code or checkpoints.

---

## Interface adequacy assessment

### Current `ContentEncoder` interface

```python
class ContentEncoder(ABC):
    def load(self, checkpoint_path: str) -> None: ...
    def encode(self, audio: np.ndarray) -> np.ndarray: ...
    def reset(self) -> None: ...
```

### Assessment

| Requirement | Satisfied by interface? | Notes |
|---|---|---|
| Load any backend from checkpoint | ✅ `load()` | |
| Accept canonical 16 kHz float32 | ✅ by doc contract | Concrete must validate |
| Return opaque np.ndarray | ✅ | Pipeline treats as opaque |
| Support streaming (stateful) | ✅ `reset()` | Concrete manages KV cache |
| Support offline (stateless) | ✅ `encode()` takes full buffer | |
| Expose frame rate | ❌ **Missing** | See below |
| Expose output dimensionality | ❌ **Missing** | |
| Expose lookahead | ❌ **Missing** | |

### Required interface additions

The current interface does **not** expose metadata needed by downstream
components (e.g., the AccentTranslator needs to know the frame rate and
representation dimension).

Two options:

**Option A (preferred)**: Add read-only metadata properties to the interface.

```python
class ContentEncoder(ABC):
    @property
    @abstractmethod
    def frame_rate_hz(self) -> int: ...
    # e.g. 50 for HuBERT, model-specific for causal CNN

    @property
    @abstractmethod
    def output_dim(self) -> int: ...
    # e.g. 768 for HuBERT Base

    @property
    @abstractmethod
    def lookahead_ms(self) -> int: ...
    # 0 = causal; >0 = bounded look-ahead; -1 = non-causal (unbounded)
```

**Option B**: Pass metadata via config, document the convention.

Recommendation: implement Option A (properties on ABC) in Phase 4 before
implementing any concrete encoder.  This prevents downstream components from
hard-coding frame rates.

---

## Recommendation

### Phase 4 (offline reference): HuBERT Base via HuggingFace transformers

| Criterion | HuBERT Base |
|---|---|
| Streaming | No (offline reference only) |
| Available | Yes — `facebook/hubert-base-ls960` |
| License | Apache 2.0 |
| Dependencies | `torch`, `transformers` |
| GPU memory | ~2–4 GB VRAM (fp32); CPU possible |
| Complexity | Low for offline use |
| Frame rate | 50 Hz / 20 ms per frame |
| Output dim | 768 |
| Lookahead | Unbounded (non-causal — offline only) |

Use HuBERT Base to:
1. Validate the `ContentEncoder` interface end-to-end.
2. Establish a quality baseline for content representation.
3. Drive the offline conversion baseline (Phase 6).
4. Produce targets for distilling a streaming student encoder.

### Phase 7+ (streaming): custom causal encoder (TVTSyn-inspired)

Build a causal 1-D CNN + causal MHSA encoder from scratch, inspired by the
TVTSyn architecture.  This requires:
- Defining the architecture in detail.
- Deciding on training data (LibriSpeech + Indian English corpora).
- Implementing the VQ bottleneck.
- Training and validating the encoder independently before integrating it
  into the full pipeline.

This is a **significant research and engineering task** that should not be
started before the offline pipeline is stable.

---

## Unresolved risks

| Risk | Severity | Mitigation |
|---|---|---|
| HuBERT representations retain speaker information | Medium | Use a speaker-information suppression layer; evaluate with speaker probing |
| Indian-accented English not in LibriSpeech pretraining data | Medium | Evaluate representation quality on IITM/MUCS/L2Arctic corpora before committing |
| No public TVTSyn implementation | High | Build causal encoder from scratch; long lead time |
| VQ codebook collapse during training | Medium | Use commitment loss, codebook EMA reset |
| PyTorch/transformers version pinning | Low | Test install in clean venv; pin to tested versions |
| VRAM insufficient for HuBERT Base + downstream models simultaneously | Medium | Measure peak VRAM with all models loaded |

---

## Required dependencies (Phase 4)

```toml
# Add to pyproject.toml [project.dependencies]
torch>=2.0
transformers>=4.30
```

`torchaudio` is optional — HuBERT can be loaded and used without it via
the `transformers` library.

---

## Implementation Status & Next Tasks

- [x] **Phase 4A**: ContentEncoder ABC contract with `frame_rate_hz`, `output_dim`, `lookahead_ms` properties.
- [x] **Phase 4B**: HuBERT offline reference encoder implemented via `Wav2Vec2FeatureExtractor` and `HubertModel`.
- [x] **Phase 4B**: Unit test suite (`tests/unit/test_hubert_encoder.py`) — fast, 0 model downloads.
- [x] **Phase 4B**: Genuine integration test (`tests/integration/test_hubert_integration.py`) — loads `facebook/hubert-base-ls960`, verifies `(T, 768)` shape and determinism.
- [x] **Phase 4B**: Benchmark script (`scripts/benchmark_hubert.py`) — measured CPU/CUDA latency, RTF, actual frame counts, and peak VRAM.
- [ ] **Phase 5**: ECAPA Speaker Embedding (next pipeline stage).
- [ ] **Phase 6**: Streaming causal student / lightweight content encoder exploration.
