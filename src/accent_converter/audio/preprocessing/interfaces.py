"""
audio.preprocessing.interfaces
================================
Abstract base classes for the three audio preprocessing components:

    VoiceActivityDetector   (VAD)
    NoiseSuppressor         (NS)
    AcousticEchoCanceller   (AEC)

Design principles
-----------------
* Each interface is independent — they can be enabled, disabled, or swapped
  individually via configuration without touching pipeline code.
* VAD returns a *label per chunk* plus a timing-preserving copy of the audio
  so that the pipeline decides what to do with silent frames (e.g. gate,
  pass through, or use for noise estimation).
* AEC keeps the far-end reference signature in the interface even though the
  initial implementation is a passthrough — this locks in the correct API
  before WebRTC integration.
* ``reset()`` is mandatory on all three so streaming sessions can restart
  cleanly.
* No ML frameworks are imported here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Shared result types
# ---------------------------------------------------------------------------


@dataclass
class VADResult:
    """Result of one VAD decision on a single audio chunk.

    Attributes
    ----------
    audio : np.ndarray
        The original audio chunk, **unmodified** (timing is preserved).
        Shape: ``(N,)`` float32.
    is_speech : bool
        True if this chunk is classified as containing speech.
    energy_db : float
        Short-time energy of the chunk in dB (relative to full scale).
        Useful for diagnostics and threshold tuning.
    """

    audio: np.ndarray
    is_speech: bool
    energy_db: float


# ---------------------------------------------------------------------------
# VoiceActivityDetector
# ---------------------------------------------------------------------------


class VoiceActivityDetector(ABC):
    """Classify each audio chunk as speech or non-speech.

    The VAD **must not delete or modify audio samples**.  It annotates each
    chunk with an ``is_speech`` label so the caller can decide independently
    how to handle non-speech frames.

    Hangover logic is the responsibility of the concrete implementation:
    a short pause inside speech should not immediately flip the label to
    non-speech.

    Methods
    -------
    process(chunk) -> VADResult
        Classify one chunk.  The chunk must already be at the canonical
        16 kHz float32 format.
    reset()
        Reset internal streaming state (hangover counter, history, etc.).
        Must be safe to call at any time.
    """

    @abstractmethod
    def process(self, chunk: np.ndarray) -> VADResult:
        """Classify *chunk* as speech or non-speech.

        Parameters
        ----------
        chunk:
            1-D float32 array, canonical 16 kHz format.

        Returns
        -------
        VADResult
            Contains the original audio (unmodified), the speech/silence
            label, and the energy in dB.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset internal streaming state."""


# ---------------------------------------------------------------------------
# NoiseSuppressor
# ---------------------------------------------------------------------------


class NoiseSuppressor(ABC):
    """Apply noise suppression to an audio chunk or full buffer.

    The suppressor should reduce stationary and/or non-stationary background
    noise while preserving speech.

    Callers are responsible for deciding when to invoke the suppressor.
    Typical patterns:
    - Apply to every chunk (stationary noise).
    - Apply only after VAD confirms a speech segment.

    Methods
    -------
    process(audio) -> np.ndarray
        Apply suppression to *audio* and return the cleaned signal.
    reset()
        Reset internal state (noise profile, adaptive estimates, etc.).
    """

    @abstractmethod
    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply noise suppression to *audio*.

        Parameters
        ----------
        audio:
            1-D float32 array, canonical 16 kHz format.

        Returns
        -------
        np.ndarray
            Cleaned 1-D float32 array, same length as input.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state (noise model, adaptive filters, etc.)."""


# ---------------------------------------------------------------------------
# AcousticEchoCanceller
# ---------------------------------------------------------------------------


class AcousticEchoCanceller(ABC):
    """Remove acoustic echo by using the far-end (loudspeaker) reference signal.

    The canonical AEC implementation for this project is **browser-side
    WebRTC**.  This interface exists so that:
    (a) the pipeline slot is correctly typed and wired,
    (b) a server-side Python fallback can be plugged in later if needed.

    Far-end reference
    -----------------
    The ``process()`` method accepts an optional ``far_end`` reference array.
    When ``far_end`` is ``None``, the implementation should do the best it
    can (e.g. passthrough, or use an internally buffered reference if
    available).

    Methods
    -------
    process(near_end, far_end=None) -> np.ndarray
        Cancel echo from *near_end* using *far_end* as the reference.
    reset()
        Reset adaptive filter state.
    """

    @abstractmethod
    def process(
        self,
        near_end: np.ndarray,
        far_end: np.ndarray | None = None,
    ) -> np.ndarray:
        """Cancel echo from *near_end* using *far_end* as the reference.

        Parameters
        ----------
        near_end:
            1-D float32 array — microphone signal (contains speech + echo).
        far_end:
            1-D float32 array — loudspeaker/remote signal (echo reference),
            or ``None`` if not available.

        Returns
        -------
        np.ndarray
            Echo-cancelled 1-D float32 array, same length as *near_end*.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset adaptive filter / delay-buffer state."""
