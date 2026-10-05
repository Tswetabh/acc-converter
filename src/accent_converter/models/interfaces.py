"""
models.interfaces
=================
Abstract base classes (ABCs) for all model components in the accent-converter
pipeline.

Every concrete model **must** subclass the appropriate interface from this
module.  The pipeline code imports only from here — never from a concrete
backend module.

Five-component canonical architecture
--------------------------------------
The pipeline has five conceptual stages:

::

    ContentEncoder        – audio waveform → content representation
    AccentTranslator      – content rep + target accent → target-accent rep
    [AcousticDecoder]     – target-accent rep + speaker embedding → mel spectrogram
    [Vocoder]             – mel spectrogram → waveform
    SpeakerEncoder        – reference audio → speaker embedding (192-d, ECAPA)

The AcousticDecoder and Vocoder are **not** separate ABCs at this stage.
They are internal stages encapsulated inside a single ``SpeechSynthesizer``
implementation.  The pipeline calls only::

    SpeechSynthesizer.synthesize(target_rep, speaker_embedding) → waveform

and the concrete backend (e.g. ``HifiGanSynthesizer``) internally chains:

::

    target_rep + speaker_embedding
          ↓
    Acoustic Decoder / Mel Predictor   (learned — produces mel spectrogram)
          ↓
    mel spectrogram
          ↓
    Neural Vocoder (e.g. HiFi-GAN)     (produces waveform from mel)
          ↓
    16 kHz float32 waveform

Standard HiFi-GAN is a mel → waveform vocoder.  It does NOT accept the
AccentTranslator output representation directly.  The intermediate
Acoustic Decoder / Mel Predictor is therefore a required component of any
concrete ``SpeechSynthesizer`` that uses a mel-based vocoder.

Locked data flow
----------------
::

    ContentEncoder.encode(audio)                           → content_rep
    AccentTranslator.translate(content_rep, target_accent) → target_rep
    SpeechSynthesizer.synthesize(target_rep,               → waveform
                                 speaker_embedding)

Speaker embedding is passed to the Synthesizer (not the Translator).
Speaker identity conditioning is applied inside the Acoustic Decoder.

ABC definitions
---------------
ContentEncoder      – audio waveform → content representation
SpeakerEncoder      – reference audio → speaker embedding
AccentTranslator    – content representation → accent-converted representation
SpeechSynthesizer   – (target rep + speaker embedding) → waveform
                      [internally: AcousticDecoder → mel → Vocoder]

Design principles
-----------------
* Each interface declares only the methods the *pipeline* calls.
* ``load(checkpoint_path)`` is kept on the interface so the pipeline/factory
  can drive loading without knowing the backend.
* ``reset()`` is mandatory: every stateful component must be resettable so
  that streaming sessions can be restarted cleanly.
* Method signatures use ``np.ndarray`` for audio and representations.
  Concrete backends may work internally with tensors; they must convert at
  their boundaries.
* No ML framework imports here — this file must import in any environment.

ContentEncoder metadata properties
-----------------------------------
Three read-only properties are required on every ContentEncoder backend:

* ``frame_rate_hz`` : int
    Frames per second of the output representation.
    e.g. 50 for HuBERT (20 ms / frame).

* ``output_dim`` : int
    Dimensionality of each output frame vector.
    e.g. 768 for HuBERT Base.

* ``lookahead_ms`` : int
    Look-ahead in milliseconds.
    0   = strictly causal.
    >0  = bounded look-ahead (e.g. 80 for TVTSyn causal encoder).
    -1  = sentinel: non-causal / unbounded (offline models only).

These properties allow downstream components (AccentTranslator, pipeline
scheduler) to query encoder metadata without knowing the concrete type.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# ContentEncoder
# ---------------------------------------------------------------------------


class ContentEncoder(ABC):
    """Convert a (normalised, 16 kHz, float32) audio array to a content
    representation.

    The representation format (shape, dtype, frame rate) is backend-specific
    and is exposed through read-only metadata properties.  The pipeline treats
    the returned array as opaque ``np.ndarray``.

    Methods
    -------
    load(checkpoint_path)
        Load model weights.  Must be called before ``encode``.
    encode(audio) -> np.ndarray
        Encode one audio chunk or full utterance.
    reset()
        Clear any internal streaming state (e.g. causal KV cache).
        Must be called between independent utterances/sessions.

    Properties
    ----------
    frame_rate_hz : int
        Output frames per second.  e.g. 50 for HuBERT.
    output_dim : int
        Dimensionality of each output frame.  e.g. 768 for HuBERT Base.
    lookahead_ms : int
        Look-ahead in ms.  0 = causal, >0 = bounded lookahead,
        -1 = non-causal / unbounded (offline only).
    """

    @abstractmethod
    def load(self, checkpoint_path: str) -> None:
        """Load model weights from *checkpoint_path*."""

    @abstractmethod
    def encode(self, audio: np.ndarray) -> np.ndarray:
        """Encode *audio* (1-D float32, 16 kHz) → content representation.

        Parameters
        ----------
        audio:
            1-D float32 array, canonical 16 kHz format.

        Returns
        -------
        np.ndarray
            Content representation of shape ``(T_frames, output_dim)``.
            ``T_frames ≈ len(audio) / sample_rate * frame_rate_hz``.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset internal streaming state.  Must be safe to call at any time."""

    @property
    @abstractmethod
    def frame_rate_hz(self) -> int:
        """Output frames per second.

        HuBERT Base/Large: 50 Hz (one frame per 20 ms).
        Causal CNN encoders: depends on architecture.
        """

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimensionality of each output representation vector.

        HuBERT Base: 768.
        HuBERT Large: 1024.
        TVTSyn VQ bottleneck: 8 (quantised) or higher (pre-VQ).
        """

    @property
    @abstractmethod
    def lookahead_ms(self) -> int:
        """Look-ahead in milliseconds.

        Returns
        -------
        int
            * ``-1`` — non-causal / unbounded (offline-only models such as
              standard HuBERT, WavLM).  These models must NOT be used in the
              live streaming path.
            * ``0`` — strictly causal; no future context.
            * ``> 0`` — bounded look-ahead, e.g. 80 ms for TVTSyn causal
              encoder.
        """


# ---------------------------------------------------------------------------
# SpeakerEncoder
# ---------------------------------------------------------------------------


class SpeakerEncoder(ABC):
    """Extract a fixed-dimensional speaker embedding from reference audio.

    The embedding is typically computed once from an enrollment recording and
    cached by the caller for reuse across chunks.

    Methods
    -------
    load(checkpoint_path)
        Load model weights.
    encode(reference_audio) -> np.ndarray
        Compute a speaker embedding vector.
    reset()
        Clear any internal state (usually a no-op for speaker encoders, but
        required by the interface for uniformity).
    """

    @abstractmethod
    def load(self, checkpoint_path: str) -> None:
        """Load model weights from *checkpoint_path*."""

    @abstractmethod
    def encode(self, reference_audio: np.ndarray) -> np.ndarray:
        """Compute a speaker embedding from *reference_audio*.

        Parameters
        ----------
        reference_audio:
            1-D float32 array, canonical 16 kHz format.

        Returns
        -------
        np.ndarray
            1-D speaker embedding vector; dimensionality is backend-specific.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal state."""


# ---------------------------------------------------------------------------
# AccentTranslator
# ---------------------------------------------------------------------------


class AccentTranslator(ABC):
    """Transform a content representation from source accent to target accent.

    The translator operates on the representation produced by a
    ``ContentEncoder``.  It must not perform ASR, TTS, or speaker conversion.
    Its sole responsibility is accent/pronunciation transformation.

    Speaker identity is **not** a responsibility of the translator.  The
    speaker embedding (from ``SpeakerEncoder``) is passed separately to the
    ``SpeechSynthesizer``, which is the component that applies speaker
    conditioning when generating the final waveform.

    Data flow::

        ContentEncoder.encode(audio)
              ↓ content_rep
        AccentTranslator.translate(content_rep, target_accent)
              ↓ target_rep
        SpeechSynthesizer.synthesize(target_rep, speaker_embedding)
              ↓ waveform

    Methods
    -------
    load(checkpoint_path)
        Load model weights.
    translate(content_rep, target_accent) -> np.ndarray
        Produce an accent-converted representation.
    reset()
        Clear internal streaming state (e.g. causal KV-cache).
    """

    @abstractmethod
    def load(self, checkpoint_path: str) -> None:
        """Load model weights from *checkpoint_path*."""

    @abstractmethod
    def translate(
        self,
        content_rep: np.ndarray,
        target_accent: str,
    ) -> np.ndarray:
        """Translate *content_rep* toward *target_accent*.

        Parameters
        ----------
        content_rep:
            Output of ``ContentEncoder.encode()``.  The exact shape and dtype
            are backend-specific and depend on the content representation
            selected during Phase 6B research.
        target_accent:
            Target accent identifier, e.g. ``"us"``, ``"uk"``, ``"neutral"``.

        Returns
        -------
        np.ndarray
            Accent-converted representation; shape is backend-specific.
            Passed directly to ``SpeechSynthesizer.synthesize()``.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset internal streaming state (e.g. causal KV-cache)."""


# ---------------------------------------------------------------------------
# SpeechSynthesizer
# ---------------------------------------------------------------------------


class SpeechSynthesizer(ABC):
    """Convert an accent-translated representation into a speech waveform.

    This is the *only* waveform synthesis interface the pipeline may use.
    The pipeline treats this as a black box::

        waveform = synthesizer.synthesize(target_rep, speaker_embedding)

    Internal architecture
    ----------------------
    Concrete implementations of this class are **composites**.  They
    internally pipeline two stages:

    ::

        target_rep  (output of AccentTranslator)
             +
        speaker_embedding  (output of SpeakerEncoder, 192-d ECAPA)
             ↓
        Acoustic Decoder / Mel Predictor
            (learned; maps target-accent representation to mel spectrogram;
             applies speaker conditioning here)
             ↓
        mel spectrogram  (e.g. 80-band log-mel)
             ↓
        Neural Vocoder  (e.g. HiFi-GAN, UnivNet, BigVGAN)
            (maps mel spectrogram to 16 kHz waveform)
             ↓
        16 kHz float32 waveform

    Standard HiFi-GAN is a mel → waveform vocoder.  It does **not** accept
    the AccentTranslator output directly.  The Acoustic Decoder / Mel
    Predictor is therefore a mandatory internal stage in any mel-based
    ``SpeechSynthesizer`` backend.

    If a future backend produces waveforms directly without a mel intermediate
    (e.g. a diffusion-based end-to-end model), it may omit the mel stage
    internally while still satisfying this interface.

    Methods
    -------
    load(checkpoint_path)
        Load model weights / config.  Must be called before ``synthesize``.
    synthesize(converted_rep, speaker_embedding) -> np.ndarray
        Generate a waveform from a target-accent representation.
    reset()
        Reset any internal autoregressive / streaming state.  Must be called
        between independent utterances or sessions.
    """

    @abstractmethod
    def load(self, checkpoint_path: str) -> None:
        """Load model weights from *checkpoint_path*."""

    @abstractmethod
    def synthesize(
        self,
        converted_rep: np.ndarray,
        speaker_embedding: np.ndarray,
    ) -> np.ndarray:
        """Synthesize a waveform from *converted_rep* and *speaker_embedding*.

        Parameters
        ----------
        converted_rep:
            Output of ``AccentTranslator.translate()``.  The exact shape
            depends on the content representation selected in Phase 6B.
            This is the *target-accent representation*, not a mel spectrogram.
            The implementation is responsible for converting it to a mel
            spectrogram internally via its Acoustic Decoder before passing
            it to the vocoder.
        speaker_embedding:
            Output of ``SpeakerEncoder.encode()``.  Shape: (192,) float32.
            Applied as speaker conditioning inside the Acoustic Decoder.

        Returns
        -------
        np.ndarray
            1-D float32 waveform at the canonical 16 kHz sample rate.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset internal autoregressive / streaming state.

        Must be called at the start of each new utterance or session to
        prevent context bleed between independent audio streams.
        """
