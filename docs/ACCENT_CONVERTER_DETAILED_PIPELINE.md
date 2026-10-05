# Real-Time Accent Converter — Detailed Pipeline & Implementation Flow

> **Document purpose:** Engineering-level description of the complete real-time accent-conversion pipeline, the role of every component, model interfaces, tensor/data flow, training strategy, implementation order, and the transition from offline validation to real-time streaming.
>
> **Primary task:** Indian English speech → US / UK / Neutral English speech
>
> **Current project state:** Audio Foundation, Audio Preprocessing, model ABCs/factory, and HuBERT offline reference encoder are implemented. The real Accent Translator, streaming Content Encoder, streaming Synthesizer, output stitching, concurrent pipeline, and browser integration remain to be implemented.

---

# 1. System Goal

The system converts a speaker's Indian-accented English into a selected target accent while preserving:

- **Linguistic content** — what the speaker said
- **Speaker identity** — who is speaking
- **Target accent characteristics** — how the speech should be pronounced
- **Naturalness** — the result should sound like continuous human speech
- **Low latency** — conversion must happen continuously during a live interaction

Supported target accents are:

```text
Indian English → US English
Indian English → UK English
Indian English → Neutral English
```

The system is intended for a live browser voice interaction, so it must operate as a **streaming system** rather than recording an entire utterance and processing it after the speaker finishes.

---

# 2. Core Design Principle

The project separates four different responsibilities:

```text
CONTENT ENCODER
    ↓
WHAT was said?

ACCENT TRANSLATOR
    ↓
HOW should it be pronounced?

SPEAKER ENCODER
    ↓
WHO is speaking?

SYNTHESIZER
    ↓
TURN the representation back into sound
```

This separation is fundamental.

The Accent Translator is the central learned transformation responsible for changing the pronunciation/accent representation. The other components make it possible to perform that transformation while preserving content and speaker identity.

---

# 3. Complete End-to-End Pipeline

```text
                    ┌──────────────────────┐
                    │   Browser Microphone  │
                    └──────────┬───────────┘
                               │
                               │ WebRTC / WebSocket
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                     AUDIO FOUNDATION                        │
│                                                             │
│ Raw audio                                                   │
│   ↓                                                         │
│ Resampler → 16 kHz                                         │
│   ↓                                                         │
│ Normalizer → mono float32 [-1, 1]                           │
│   ↓                                                         │
│ Chunker → 80 ms frames (1,280 samples)                      │
│   ↓                                                         │
│ ContextManager → 640 ms bounded lookback                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   AUDIO PREPROCESSING                       │
│                                                             │
│ AEC → currently passthrough                                │
│   ↓                                                         │
│ Noise Suppression → spectral gating                         │
│   ↓                                                         │
│ VAD → speech/non-speech decision                            │
└──────────────────────────┬──────────────────────────────────┘
                           │
                     speech frames
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      CONTENT ENCODER                        │
│                                                             │
│ Development/reference:                                     │
│   HuBERT Base                                              │
│   output: (T, 768)                                         │
│   non-causal / offline only                                │
│                                                             │
│ Final streaming path:                                      │
│   TVTSyn-style causal encoder                              │
│   output: (T, D)                                           │
│   causal CNN + MHSA + VQ                                   │
│   ring KV-cache                                             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           │ content / phonetic representation
                           ▼
                  ┌──────────────────────┐
                  │   ACCENT TRANSLATOR  │
                  │                      │
                  │ Causal ConvNeXt      │
                  │        ↓             │
                  │ Causal Transformer   │
                  │        ↓             │
                  │ Causal ConvNeXt      │
                  │                      │
                  │ Conditioning:        │
                  │ target accent        │
                  │                      │
                  │ Output:              │
                  │ target-accent        │
                  │ representation       │
                  └──────────┬───────────┘
                             │
                             ▼
                  target-accent representation
                             │
                             │
                 ┌───────────┴───────────┐
                 │                       │
                 │                       │
                 ▼                       ▼
       ┌─────────────────┐    ┌────────────────────────┐
       │ ECAPA-TDNN      │    │ Target accent ID       │
       │                 │    │ / embedding            │
       │ speaker         │    │                        │
       │ embedding       │    │ US / UK / Neutral      │
       │ (192,)          │    │                        │
       └────────┬────────┘    └──────────┬─────────────┘
                │                        │
                └────────────┬───────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   WAVEFORM SYNTHESIZER                      │
│                                                             │
│ Input:                                                       │
│   target-accent representation                               │
│   + speaker embedding                                        │
│                                                             │
│ Candidate backends:                                         │
│   HiFi-GAN                                                   │
│   UnivNet                                                    │
│   BigVGAN                                                    │
│   TVTSyn decoder                                             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
                converted 16 kHz waveform
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      OUTPUT STITCHING                        │
│                                                             │
│ overlap-add / crossfade                                     │
│ remove chunk boundaries / clicks                            │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           │ WebRTC / WebSocket
                           ▼
              ┌──────────────────────────┐
              │ Browser Playback / Call  │
              └──────────────────────────┘
```

---

# 4. Audio Contract

Every internal audio module uses a single canonical format:

```text
Sample rate:       16,000 Hz
Channels:          1 (mono)
dtype:             float32
Amplitude:         approximately [-1, 1]
Chunk size:        80 ms
Samples/chunk:     1,280
Lookback:          640 ms
Frames of lookback: 8
```

## 4.1 Why 16 kHz?

The speech models used in the project operate on 16 kHz audio.

At 16,000 samples/second:

```text
1 second = 16,000 samples

80 ms = 0.08 × 16,000
     = 1,280 samples
```

Therefore one streaming audio chunk is:

```text
audio_chunk.shape == (1280,)
```

for mono audio.

---

# 5. Stage 1 — Browser Audio Capture

## Goal

Capture microphone audio continuously.

Browser-side capture is expected to use:

```javascript
navigator.mediaDevices.getUserMedia({
    audio: true
})
```

The browser produces an audio stream that is transmitted to the backend using either:

```text
WebRTC
```

or

```text
WebSocket
```

### Important separation

WebRTC/WebSocket is the **transport layer**.

It is not the accent-conversion model.

```text
Browser mic
   ↓
transport
   ↓
Python/ML pipeline
```

---

# 6. Stage 2 — Audio Foundation

The Audio Foundation converts arbitrary input into the project's canonical representation.

## 6.1 Resampler

```text
Input:
    any reasonable source sample rate

Output:
    16 kHz
```

The project uses polyphase rational resampling.

Conceptually:

```text
48 kHz microphone
       ↓
resampler
       ↓
16 kHz
```

---

## 6.2 Normalizer

The normalizer performs:

```text
multi-channel audio
       ↓
mono

integer / other dtype
       ↓
float32

amplitude
       ↓
approximately [-1, 1]
```

Example:

```text
int16:
[-32768, 32767]

        ↓

float32:
[-1.0, 1.0]
```

This provides a predictable input representation for the ML models.

---

## 6.3 Chunker

The stream is divided into fixed-size frames:

```text
80 ms = 1,280 samples
```

Therefore:

```text
continuous waveform

─────────────────────────────────────────────→ time

|------80ms------|------80ms------|------80ms------|
     chunk 1            chunk 2            chunk 3
```

The system must process these chunks continuously.

---

## 6.4 Context Manager

The ContextManager keeps a bounded lookback:

```text
640 ms = 8 × 80 ms
```

Conceptually:

```text
current frame
     ↓
[oldest ... recent context ... current]
     ← 640 ms →
```

This allows downstream models to receive temporal context without keeping unbounded audio history.

---

# 7. Stage 3 — Audio Preprocessing

## 7.1 VAD

Voice Activity Detection decides whether the incoming audio contains speech.

Current implementation:

```text
EnergyVAD
```

It uses:

- short-time energy
- configurable threshold
- hangover counter

Example:

```text
speech      speech      speech      silence
  1           1           1            0
```

Non-speech frames can be prevented from entering expensive model inference.

---

## 7.2 Noise Suppression

Current backend:

```text
noisereduce
```

with spectral gating.

Purpose:

```text
microphone audio
       ↓
background noise reduction
       ↓
cleaner speech
```

This is important because downstream neural models should not be forced to interpret avoidable background noise as speech.

---

## 7.3 AEC

Current implementation:

```text
PassthroughAEC
```

This is a placeholder.

The intended real solution is browser-side WebRTC AEC3 during browser integration.

AEC is needed when the speaker's microphone may pick up audio coming from the other participant.

---

# 8. Stage 4 — Content Encoder

The Content Encoder converts waveform audio into a learned representation of speech content.

Its purpose is to emphasize:

```text
WHAT was said
```

rather than directly carrying the full waveform.

---

# 9. HuBERT Offline Reference Encoder

The currently implemented backend is:

```text
facebook/hubert-base-ls960
```

It produces:

```text
(T, 768)
```

float32 features.

## 9.1 Meaning of T

`T` is the number of output time steps.

For 1 second of 16 kHz audio:

```text
N = 16,000 samples
```

The implemented HuBERT frontend uses:

```text
cumulative stride = 320 samples
             = 20 ms

receptive field = 400 samples
                = 25 ms
```

The frame-count formula is:

```text
T = floor((N - 400) / 320) + 1
```

For one second:

```text
T = floor((16000 - 400) / 320) + 1
  = 49
```

Therefore:

```text
1 sec audio
    ↓
HuBERT
    ↓
(49, 768)
```

This means:

```text
49 time steps
×
768-dimensional feature vector per time step
```

The 768 values are learned features; individual dimensions should not be interpreted as one explicit phoneme or one explicit accent attribute.

---

# 10. Why HuBERT Is Not the Final Streaming Encoder

HuBERT Base is non-causal.

Therefore it is suitable for:

```text
offline reference
development
benchmarking
baseline experiments
```

but not the final live pipeline.

The project therefore uses a two-tier strategy:

```text
Phases 4–6
    ↓
HuBERT offline reference

Phase 7+
    ↓
custom causal streaming encoder
```

HuBERT is not intended to run as the live streaming content encoder.

---

# 11. Stage 4 Final Target — TVTSyn-Style Causal Content Encoder

The planned streaming content encoder is:

```text
Input waveform
      ↓
Causal 1-D CNN frontend
      ↓
causal MHSA blocks
      ↓
VQ bottleneck
      ↓
content / phonetic representation
```

Current architecture target:

```text
Causal CNN
     ↓
8 × causal Multi-Head Self-Attention
     ↓
ring KV-cache
     ↓
VQ bottleneck
```

Target constraints:

```text
lookahead <= 80 ms
attention history ≈ 2 seconds
```

With 80 ms chunks, approximately:

```text
2 seconds / 80 ms = 25 chunks
```

of recent context can be represented in the target attention window.

The custom encoder must be trained from scratch because the project does not rely on a public checkpoint for this exact streaming architecture.

---

# 12. Continuous Features vs Discrete Content Tokens

This is a critical design decision.

HuBERT gives continuous vectors:

```text
(T, 768)
```

A VQ-style encoder can instead produce discrete content units:

```text
token_1
token_2
token_3
...
token_T
```

Conceptually:

```text
waveform
   ↓
encoder
   ↓
continuous features
   ↓
Vector Quantization
   ↓
discrete content tokens
```

Discrete or phonetic-oriented representations are attractive for the Accent Translator because the translator is easier to formulate as:

```text
source-accent pronunciation units
             ↓
target-accent pronunciation units
```

rather than requiring the translator to learn an unconstrained transformation directly in arbitrary 768-dimensional feature space.

The exact final representation must be selected and validated during the real Accent Translator research phase.

---

# 13. Stage 5 — Speaker Encoder

Current target backend:

```text
ECAPA-TDNN
speechbrain/spkrec-ecapa-voxceleb
```

Output:

```text
(192,)
```

float32 speaker embedding.

The role is:

```text
WHO is speaking?
```

not:

```text
What was said?
```

and not:

```text
Which accent should be used?
```

## 13.1 Enrollment Flow

Speaker embedding is computed once:

```text
reference/enrollment audio
        ↓
ECAPA-TDNN
        ↓
192-dimensional vector
        ↓
cache for current session
```

It does not need to be recomputed for every 80 ms audio chunk.

This saves compute and makes the runtime pipeline cleaner.

---

# 14. Stage 6 — Accent Translator

This is the central learned transformation of the project.

## 14.1 Responsibility

The Accent Translator receives a source-accent content/phonetic representation and converts it into a target-accent representation.

Conceptually:

```text
Indian-accent content representation
                +
          target accent
                ↓
         Accent Translator
                ↓
target-accent content representation
```

The goal is to preserve:

```text
linguistic content
+
speaker identity
```

while changing:

```text
pronunciation / accent realization
```

The translator should therefore not be thought of as a simple phoneme lookup table.

Accent differences can involve:

- vowel realization
- consonant realization
- timing
- stress
- rhythm
- coarticulation
- pronunciation patterns across neighboring sounds

The network must learn these patterns from speech data.

---

# 15. Current Accent Translator Stub

Current implementation:

```text
NaiveAccentTranslator
```

Its operation is:

```text
x → x
```

or:

```text
input representation
        ↓
identity function
        ↓
same representation
```

Therefore it performs **no actual accent conversion**.

Its purpose is engineering:

```text
pipeline wiring
interface testing
end-to-end plumbing
```

It must eventually be replaced by the learned translator.

---

# 16. Recommended Real Accent Translator Architecture

The intended research direction is a causal, non-autoregressive sequence model:

```text
SOURCE CONTENT TOKENS
        ↓
Causal ConvNeXt
        ↓
Causal Transformer
        ↓
Causal ConvNeXt
        ↓
Target-token prediction
```

Conceptually:

```text
                 ┌───────────────────────┐
content tokens ─→│ causal local layers  │
                 └──────────┬────────────┘
                            ↓
                 ┌───────────────────────┐
                 │ causal Transformer    │
                 │ temporal context      │
                 └──────────┬────────────┘
                            ↓
                 ┌───────────────────────┐
                 │ causal local layers   │
                 └──────────┬────────────┘
                            ↓
                    target-accent tokens
```

### Why this combination?

## ConvNeXt

Convolutional blocks are useful for local speech patterns:

```text
neighboring phonetic units
coarticulation
local pronunciation transitions
```

## Transformer

Self-attention can model broader temporal dependencies:

```text
phonetic context
word-level context
longer pronunciation dependencies
```

Using both gives the translator access to:

```text
LOCAL CONTEXT
+
LONGER-RANGE CONTEXT
```

---

# 17. Target Accent Conditioning

The translator must know which accent to produce.

The model therefore receives a target accent condition:

```text
US
UK
Neutral
```

This can be represented as:

```text
accent ID
```

or a learned:

```text
accent embedding
```

Conceptually:

```text
content tokens
      +
target accent embedding
      ↓
Accent Translator
      ↓
target-accent tokens
```

This allows the same translator architecture to support multiple target accents.

---

# 18. Does the Translator Change Phonemes?

Not necessarily in a literal one-to-one sense.

Avoid designing the system as:

```text
Indian phoneme X
       ↓
American phoneme Y
```

for every sound.

A better formulation is:

```text
source-accent realization
          ↓
learned transformation
          ↓
target-accent realization
```

The model can therefore learn changes involving:

```text
phonetic realization
coarticulation
duration
stress/rhythm
context-dependent pronunciation
```

The representation and training target determine exactly how much of this is learned by the translator versus the synthesizer.

---

# 19. Translator Inputs and Outputs

A clean abstract model interface is:

```text
translate(
    content_representation,
    target_accent
)
    ↓
target_content_representation
```

The implementation should not prematurely hard-code:

```text
(T, 768) → (T, 768)
```

because the final representation may be:

```text
continuous features
```

or:

```text
PPG features
```

or:

```text
discrete VQ tokens
```

or another phonetic/content representation selected during Phase 6.

---

# 20. Training Strategy

The Accent Translator is not something that can simply be downloaded and dropped into the pipeline.

It must learn from speech data.

A simplified supervised training example looks like:

```text
Source:
Indian-accented utterance

Target:
same linguistic content realized with the desired target accent
```

The training pair is conceptually:

```text
SOURCE AUDIO
     ↓
Content Encoder
     ↓
source representation
     ↓
Accent Translator
     ↓
predicted target representation

                     compare with

TARGET REPRESENTATION
```

---

# 21. Parallel vs Non-Parallel Data

Perfectly parallel data would look like:

```text
Indian speaker:
"I need water."

        ↕

Target-accent speech:
"I need water."
```

with very strong content correspondence.

However, collecting a large amount of perfectly matched Indian/target-accent recordings is difficult.

Therefore the project should evaluate:

```text
parallel training
```

and:

```text
weakly parallel / non-parallel approaches
```

depending on the selected representation and available datasets.

This choice is part of Accent Translator research.

---

# 22. Loss Functions

A suitable sequence-translation training objective can combine:

## 22.1 Cross-Entropy Loss

Used to train target token prediction.

Conceptually:

```text
predicted target token
        vs
correct target token
```

The model learns:

```text
source token sequence
        ↓
target pronunciation token sequence
```

---

## 22.2 CTC Loss

CTC is useful when exact input/output frame alignment is uncertain.

Instead of requiring:

```text
source frame 1 ↔ target frame 1
source frame 2 ↔ target frame 2
...
```

CTC provides sequence-level alignment flexibility.

A combined objective can therefore be represented as:

```text
L_total =
    λ1 * L_CE
  + λ2 * L_CTC
```

The exact weighting must be validated experimentally.

---

# 23. Speaker Preservation

Accent conversion must not accidentally replace the speaker.

The intended result is:

```text
same speaker identity
+
different accent
```

not:

```text
Indian speaker
      ↓
random target-accent speaker
```

Speaker preservation can be evaluated using ECAPA embeddings:

```text
ECAPA(original audio)
        ↓
speaker vector A

ECAPA(converted audio)
        ↓
speaker vector B

cosine_similarity(A, B)
```

Higher similarity indicates stronger preservation of speaker identity, subject to proper evaluation conditions.

---

# 24. Synthesizer

The Accent Translator does not directly produce the final waveform.

It produces a representation that the synthesizer can turn into audio.

Possible synthesizer backends:

```text
HiFi-GAN
UnivNet
BigVGAN
TVTSyn decoder
```

The current project lists a vocoder backend as the primary synthesis direction and a TVTSyn decoder as an alternative.

---

# 25. Speaker + Target Representation → Speech

The conceptual synthesis input is:

```text
target-accent representation
        +
speaker embedding
        ↓
Synthesizer
        ↓
16 kHz waveform
```

This is where:

```text
WHAT
+
HOW
+
WHO
```

becomes actual sound.

---

# 26. Streaming Synthesis

The synthesizer must eventually support chunk-by-chunk inference.

The problem is:

```text
chunk 1 → audio 1
chunk 2 → audio 2
chunk 3 → audio 3
```

can introduce boundaries:

```text
|click|jump|phase discontinuity|
```

Therefore the output must be stitched.

---

# 27. Stage 9 — Output Stitching

The current plan is:

```text
overlap-add
```

or:

```text
crossfade
```

Example:

```text
generated chunk A
       ────────────────
                     \
                      \ overlap
                       ────────────────
                       generated chunk B
```

The overlapping region is blended so that the result becomes:

```text
continuous waveform
```

rather than independent audio fragments.

Things to measure:

- clicks
- discontinuities
- phase problems
- volume modulation
- duplicate/overlapping speech

---

# 28. Stage 10 — Concurrent Real-Time Pipeline

A naive implementation would do:

```text
capture
  ↓
preprocess
  ↓
encode
  ↓
translate
  ↓
synthesize
  ↓
play
  ↓
capture next chunk
```

This is undesirable because a slow stage blocks everything behind it.

The final system should run conceptually as:

```text
capture || preprocessing || inference || playback
```

using:

```text
queues
ring buffers
threading
asyncio
```

The audio callback should not block on expensive ML inference.

---

# 29. Runtime Queue Design

A practical streaming architecture can use separate buffers:

```text
Mic Input Buffer
      ↓
Preprocessing Queue
      ↓
Content Queue
      ↓
Translation Queue
      ↓
Synthesis Queue
      ↓
Output Buffer
```

Each stage consumes the output from the previous stage.

This makes it possible to pipeline work.

Example:

```text
while chunk 3 is being translated:

chunk 4 → preprocessing

chunk 2 → synthesis

chunk 1 → playback
```

This is how the system approaches continuous real-time operation.

---

# 30. Cache Management

There are several different caches. They must not be confused.

## 30.1 Audio Context Cache

Stores recent raw waveform/context.

```text
640 ms lookback
```

Purpose:

```text
provide recent audio context
```

---

## 30.2 Transformer KV Cache

Stores attention keys/values from recent temporal context.

Purpose:

```text
avoid recomputing old attention states
```

Target window for the streaming content encoder:

```text
≈ 2 seconds
```

---

## 30.3 Model Cache

Stores already-loaded model weights in RAM/VRAM.

Purpose:

```text
load model once
reuse model many times
```

Models must never be loaded independently for every 80 ms chunk.

---

## 30.4 Speaker Embedding Cache

Stores:

```text
ECAPA 192-d speaker vector
```

after enrollment.

Purpose:

```text
compute once
reuse throughout session
```

---

## 30.5 Output Buffer

Stores generated audio waiting for playback.

Purpose:

```text
smooth timing differences between inference and audio playback
```

---

# 31. Latency Model

The final system has a latency target of:

```text
< 200 ms end-to-end
```

The relevant stages are:

```text
capture
+
preprocessing
+
content encoding
+
accent translation
+
synthesis
+
output buffering/stitching
+
transport/playback
```

The actual implementation must measure every stage.

Current known measurement:

```text
HuBERT CUDA, 1 second audio ≈ 11.1 ms
```

but this is an offline-reference measurement and should not be treated as the final streaming latency.

The project currently has no measured Accent Translator or vocoder latency.

---

# 32. Recommended Implementation Order

Do not build the full streaming system in one step.

Use the following progression.

---

## Phase 1 — Repository Stabilization

Completed.

Purpose:

```text
repo structure
pyproject
test infrastructure
interfaces
```

---

## Phase 2 — Audio Foundation

Completed.

Build/verify:

```text
Resampler
Normalizer
Chunker
ContextManager
```

Result:

```text
raw audio
    ↓
16 kHz mono float32
    ↓
80 ms chunks
```

---

## Phase 3 — Audio Preprocessing

Completed.

Build/verify:

```text
VAD
Noise Suppression
AEC interface
```

Result:

```text
clean speech frames
```

---

## Phase 4B — HuBERT Offline Reference

Completed.

Result:

```text
audio
 ↓
HuBERT
 ↓
(T, 768)
```

Purpose:

```text
validate model interfaces
measure content representation
establish reference behavior
```

---

# 33. Phase 5 — ECAPA Speaker Encoder

Next implementation.

Tasks:

```text
1. Install SpeechBrain dependencies
2. Implement model loading
3. Implement encode(reference_audio)
4. Validate (192,) output
5. Add integration tests
6. Benchmark enrollment latency
7. Verify deterministic embedding behavior
8. Cache embedding at session level
```

Expected output:

```text
reference_audio
      ↓
ECAPA-TDNN
      ↓
speaker_embedding.shape == (192,)
```

---

# 34. Phase 6 — Real Accent Conversion

This is the most important ML research phase.

Do not treat this as merely:

```text
HuBERT
 ↓
some neural net
 ↓
vocoder
```

Instead, explicitly resolve:

### Step 1 — Representation

Choose:

```text
continuous SSL features
```

or:

```text
PPG
```

or:

```text
discrete VQ content tokens
```

or another validated phonetic/content representation.

### Step 2 — Translator Architecture

Primary research direction:

```text
Causal ConvNeXt
      ↓
Causal Transformer
      ↓
Causal ConvNeXt
```

Alternatives remain:

```text
PPG-based
Seq2Seq
CTC-based
```

### Step 3 — Target Accent Conditioning

Support:

```text
US
UK
Neutral
```

through an accent ID or learned accent embedding.

### Step 4 — Training Data

Determine:

```text
parallel
weakly parallel
non-parallel
```

based on available corpora and target quality.

### Step 5 — Training Objective

Evaluate a combination of:

```text
Cross-Entropy
CTC
content preservation objective
speaker preservation objective where required
```

### Step 6 — Offline End-to-End Test

Produce:

```text
input.wav
   ↓
content encoder
   ↓
accent translator
   ↓
synthesizer
   ↓
converted.wav
```

The project should not proceed to complex streaming optimization until an offline conversion baseline is demonstrably working.

---

# 35. Phase 7 — Streaming Content Encoder

Only after Phase 6 has validated the representation and translation concept:

```text
replace offline HuBERT
          ↓
causal streaming content encoder
```

Target:

```text
causal CNN
+
MHSA
+
ring KV-cache
+
VQ bottleneck
```

Lookahead:

```text
<= 80 ms
```

The streaming representation must preserve enough linguistic/phonetic information for the Accent Translator to continue performing the conversion successfully.

---

# 36. Phase 8 — Streaming Waveform Generation

Integrate a streaming-compatible synthesizer.

Candidates:

```text
HiFi-GAN
UnivNet
BigVGAN
TVTSyn decoder
```

Measure:

```text
per-chunk inference time
real-time factor
VRAM usage
audio continuity
quality
```

---

# 37. Phase 9 — Output Stitching

Add:

```text
overlap-add
crossfade
```

Then evaluate:

```text
boundary clicks
continuity
timing
volume modulation
```

---

# 38. Phase 10 — Real-Time Concurrency

Move from:

```text
sequential pipeline
```

to:

```text
concurrent pipeline
```

using:

```text
capture thread/task
preprocessing worker
content worker
translator worker
synthesis worker
playback worker
```

with bounded queues between them.

The system must handle back-pressure and avoid unbounded memory growth.

---

# 39. Phase 11 — Browser Integration

Final browser integration:

```text
getUserMedia
     ↓
browser AEC3
     ↓
transport
     ↓
Python/ML backend
     ↓
converted audio
     ↓
transport
     ↓
browser playback
```

At this point the project becomes a real end-to-end live accent conversion system.

---

# 40. Model Interfaces

The project uses abstract interfaces so the pipeline does not depend directly on one concrete model.

Conceptually:

```python
class ContentEncoder:
    load(...)
    encode(audio)
```

```python
class SpeakerEncoder:
    load(...)
    encode(reference_audio)
```

```python
class AccentTranslator:
    load(...)
    translate(content_representation, target_accent)
```

```python
class SpeechSynthesizer:
    load(...)
    synthesize(content_representation, speaker_embedding)
```

This allows the backend to change without rewriting the pipeline.

---

# 41. Recommended Accent Translator Interface

The translator interface should be designed around the actual responsibility:

```python
translate(
    content_representation,
    target_accent
) -> target_representation
```

A future implementation may additionally require:

```python
translate(
    content_representation,
    target_accent,
    state/cache=None
) -> (target_representation, state)
```

for streaming inference.

The streaming implementation should be stateful where required for:

```text
attention KV-cache
convolution state
temporal context
```

while the high-level pipeline remains backend-independent.

---

# 42. Recommended Data Flow With Shapes

A simplified offline example:

```text
Input audio:
    shape = (N,)

Example:
    N = 16,000 for 1 second
```

HuBERT:

```text
(N,)
   ↓
HubERT
   ↓
(T, 768)

For 1 second:
(49, 768)
```

ECAPA:

```text
reference audio
   ↓
ECAPA
   ↓
(192,)
```

Accent Translator:

```text
content representation
+
target accent condition
   ↓
target-accent representation
```

Synthesizer:

```text
target-accent representation
+
speaker embedding (192,)
   ↓
waveform
```

The exact translator tensor dimensions are intentionally not frozen until the representation is selected in Phase 6.

---

# 43. Example: One Utterance Through the System

Suppose the speaker says:

```text
"Could you send me the report?"
```

The conceptual flow is:

```text
1. Microphone
   ↓
2. Resample to 16 kHz
   ↓
3. Normalize to mono float32
   ↓
4. Split into 80 ms chunks
   ↓
5. Add temporal context
   ↓
6. VAD confirms speech
   ↓
7. Noise suppression
   ↓
8. Content Encoder
   ↓
   content / phonetic representation
   ↓
9. Accent Translator
   target = US
   ↓
   target-US representation
   ↓
10. ECAPA speaker embedding
    (precomputed during enrollment)
   ↓
11. Synthesizer
   ↓
12. output audio chunks
   ↓
13. overlap-add / crossfade
   ↓
14. browser playback
```

The words remain:

```text
"Could you send me the report?"
```

while the pronunciation characteristics are transformed toward the selected target accent.

---

# 44. What Each Model Should NOT Do

## Content Encoder should NOT:

```text
convert accents
```

It should primarily provide a content-oriented representation.

## ECAPA should NOT:

```text
convert accents
```

It represents speaker identity.

## Accent Translator should NOT:

```text
generate raw waveform directly
```

unless the chosen architecture explicitly combines translation and synthesis.

In the modular project architecture, its output should be an intermediate representation.

## Synthesizer should NOT:

```text
decide what the speaker said
```

It should realize the provided representation as audio.

---

# 45. What Makes the Accent Translator the USP

The system's technical novelty/central capability is not:

```text
WebRTC
```

or:

```text
16 kHz resampling
```

or:

```text
ECAPA-TDNN
```

Those are supporting technologies.

The central conversion problem is:

```text
Indian-accented speech
          ↓
content-preserving accent transformation
          ↓
US / UK / Neutral speech
```

The Accent Translator is the component that performs the learned transformation between pronunciation domains.

The project's engineering challenge is then to make that transformation:

```text
accurate
+
speaker-preserving
+
content-preserving
+
streaming
+
low-latency
```

---

# 46. Evaluation Plan

Evaluation should be separated by objective.

## Content Preservation

Use:

```text
ASR → WER
```

Compare:

```text
source speech transcript
vs
converted speech transcript
```

Goal:

```text
conversion should not change the words
```

---

## Accent Reduction

Use:

```text
accent classifier
+
human listening evaluation
```

The classifier should distinguish source vs target accent characteristics, while human evaluation measures perceived accent conversion quality.

---

## Speaker Similarity

Use:

```text
ECAPA embeddings
+
cosine similarity
```

Goal:

```text
converted speaker should still sound like original speaker
```

---

## Naturalness

Use:

```text
UTMOS / MOS-style objective and human listening
```

---

## Streaming Performance

Measure:

```text
real-time factor
per-stage latency
queue delay
buffer delay
end-to-end latency
```

---

## Stability

Run long sessions:

```text
10-minute streaming test
```

and monitor:

```text
dropouts
NaN values
queue growth
buffer underflow
memory growth
GPU memory growth
audio discontinuities
```

---

# 47. Critical Engineering Risks

## Risk 1 — Representation mismatch

If:

```text
Content Encoder output
```

is not compatible with:

```text
Accent Translator input
```

the pipeline cannot be trained effectively.

---

## Risk 2 — Translator changes content

The model might produce better accent characteristics while increasing WER.

This is why content preservation must be measured explicitly.

---

## Risk 3 — Speaker identity degradation

The converted audio may sound target-accented but like a different speaker.

This is why ECAPA similarity is measured.

---

## Risk 4 — Offline model works but streaming model fails

A model that works on complete utterances may depend on future context.

The streaming version must respect:

```text
causality
+
bounded lookahead
+
bounded context
```

---

## Risk 5 — Vocoder becomes the latency bottleneck

A low-latency translator is useless if synthesis adds hundreds of milliseconds.

Each component must be benchmarked independently.

---

## Risk 6 — Chunk boundary artifacts

Independent synthesis of 80 ms chunks can introduce audible discontinuities.

This requires explicit output stitching and/or a streaming decoder designed for continuous generation.

---

# 48. Final Target Architecture

The final system can be summarized as:

```text
                    LIVE AUDIO
                        │
                        ▼
                 Browser Mic
                        │
                WebRTC / WebSocket
                        │
                        ▼
             ┌───────────────────┐
             │ Audio Foundation  │
             │ 16 kHz / mono     │
             │ 80 ms chunks      │
             │ 640 ms lookback   │
             └────────┬──────────┘
                      │
                      ▼
             ┌───────────────────┐
             │ Preprocessing     │
             │ VAD + NS + AEC    │
             └────────┬──────────┘
                      │
                      ▼
             ┌───────────────────┐
             │ Content Encoder   │
             │ causal streaming  │
             │ content/phonetic  │
             │ representation    │
             └────────┬──────────┘
                      │
                      ▼
             ┌───────────────────┐
             │ Accent Translator │
             │                   │
             │ ConvNeXt          │
             │      ↓            │
             │ Transformer       │
             │      ↓            │
             │ ConvNeXt          │
             │                   │
             │ Indian → target   │
             └────────┬──────────┘
                      │
                      │
             ┌────────┴─────────┐
             │                  │
             ▼                  ▼
       target accent       ECAPA speaker
       representation      embedding
             │                  │
             └────────┬─────────┘
                      ▼
             ┌───────────────────┐
             │   Synthesizer     │
             │                   │
             │ HiFi-GAN / etc.   │
             └────────┬──────────┘
                      │
                      ▼
             ┌───────────────────┐
             │ Output Stitching  │
             │ overlap-add       │
             │ / crossfade       │
             └────────┬──────────┘
                      │
                      ▼
                 Browser Audio
```

---

# 49. Implementation Rule of Thumb

Do not optimize the system in the wrong order.

The recommended dependency order is:

```text
Audio correctness
      ↓
Content representation
      ↓
Accent Translator
      ↓
Offline end-to-end conversion
      ↓
Speaker preservation
      ↓
Streaming content encoder
      ↓
Streaming synthesizer
      ↓
Output stitching
      ↓
Concurrent execution
      ↓
Browser integration
      ↓
Latency optimization
```

The most important milestone before the final streaming work is:

```text
Indian speech
      ↓
content representation
      ↓
REAL Accent Translator
      ↓
target-accent representation
      ↓
synthesis
      ↓
AUDIBLE TARGET-ACCENT SPEECH
```

Once this works offline, the rest of the project is primarily about making the same conversion **causal, continuous, low-latency, and robust**.

---

# 50. Current Status vs Final Target

| Component | Current | Final target |
|---|---|---|
| Browser capture | Not started | getUserMedia |
| Transport | Not started | WebRTC / WebSocket |
| Resampler | Implemented | 16 kHz |
| Normalizer | Implemented | mono float32 |
| Chunker | Implemented | 80 ms |
| Context manager | Implemented | 640 ms lookback |
| VAD | Implemented | EnergyVAD initially |
| Noise suppression | Implemented | noisereduce initially |
| AEC | Passthrough | browser WebRTC AEC3 |
| Content Encoder | HuBERT offline | causal TVTSyn-style |
| Speaker Encoder | Planned | ECAPA-TDNN |
| Accent Translator | Identity stub | learned neural translator |
| Translator architecture | TBD | causal ConvNeXt + Transformer direction |
| Accent conditioning | TBD | US / UK / Neutral |
| Translator representation | TBD | selected during Phase 6 |
| Translator loss | TBD | CE + CTC direction |
| Synthesizer | Stub | streaming neural vocoder / TVTSyn decoder |
| Output stitching | Not started | overlap-add / crossfade |
| Concurrency | Not started | queues + workers |
| Browser playback | Not started | low-latency live audio |
| End-to-end latency | Not measured | < 200 ms target |

---

# 51. Final Mental Model

The simplest way to remember the architecture is:

```text
Audio
  ↓
CONTENT ENCODER
"What did they say?"
  ↓
CONTENT / PHONETIC REPRESENTATION
  ↓
ACCENT TRANSLATOR
"How should they pronounce it?"
  ↓
TARGET-ACCENT REPRESENTATION
  +
SPEAKER EMBEDDING
"Who should sound like this?"
  ↓
SYNTHESIZER
"Turn it into sound."
  ↓
Converted Audio
```

The project is therefore not simply:

```text
Indian audio → neural network → American audio
```

It is a modular transformation:

```text
audio
  → content
  → target-accent transformation
  → speaker-conditioned synthesis
  → continuous audio
```

with the final engineering constraint that every step must operate fast enough for live conversation.
