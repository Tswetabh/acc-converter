"""
audio.preprocessing.vad.energy
================================
Energy-based Voice Activity Detector (VAD) with hangover.

Algorithm
---------
1. Compute short-time RMS energy of each 80 ms chunk.
2. Convert to dB (floor at -96 dB).
3. Compare against a configurable ``threshold_db``.
4. Apply **hangover**: once speech is detected, keep ``is_speech=True``
   for at least ``hangover_frames`` additional frames before flipping back
   to silence.  This prevents rapid toggling on brief pauses.

Why energy-based?
-----------------
``webrtcvad`` requires a C extension that cannot be compiled on this
machine (no MSVC on Python 3.14).  A properly tuned energy VAD with
hangover is accurate enough for the accent-converter use case where the
goal is to gate model processing, not to produce reference-quality
speech/silence annotations.

Limitations
-----------
* Sensitive to very loud non-speech noise (e.g. music, fan, keyboard).
* Does not model spectral shape — this is a deliberate trade-off to avoid
  ML dependencies.
* Should be replaced with a WebRTC or ML-based VAD when browser integration
  is available (Phase 11).

Backend name: ``"energy"``

Configuration
-------------
::

    preprocessing:
      vad:
        backend: energy
        threshold_db: -40      # energy below this → silence  (default: -40)
        hangover_frames: 8     # frames to hold speech after last detection
                               #   8 × 80 ms = 640 ms hangover (default)
"""

from __future__ import annotations

import numpy as np

from accent_converter.audio.preprocessing.interfaces import (
    VoiceActivityDetector,
    VADResult,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLD_DB: float = -40.0   # dB below full scale
DEFAULT_HANGOVER_FRAMES: int = 8       # × 80 ms = 640 ms

_FLOOR_DB: float = -96.0              # dB floor for silent/zero signal


# ---------------------------------------------------------------------------
# EnergyVAD
# ---------------------------------------------------------------------------


class EnergyVAD(VoiceActivityDetector):
    """Short-time energy VAD with hangover counter.

    Parameters
    ----------
    threshold_db:
        Energy threshold in dB relative to full scale (float32 peak = 0 dB).
        Chunks with energy below this threshold are classified as non-speech.
        Default: ``-40.0`` dB.
    hangover_frames:
        Number of consecutive non-speech frames to hold a ``is_speech=True``
        label after the last speech-positive frame.  This prevents rapid
        toggling during brief pauses.  Default: 8 (640 ms at 80 ms chunks).

    Attributes
    ----------
    threshold_db : float
    hangover_frames : int
    _hangover_counter : int
        Counts down from ``hangover_frames`` after each non-speech detection.
        When > 0, ``is_speech`` is returned as ``True`` despite the energy
        being below threshold.
    """

    BACKEND_NAME: str = "energy"

    def __init__(
        self,
        threshold_db: float = DEFAULT_THRESHOLD_DB,
        hangover_frames: int = DEFAULT_HANGOVER_FRAMES,
    ) -> None:
        if hangover_frames < 0:
            raise ValueError(
                f"hangover_frames must be >= 0, got {hangover_frames}"
            )
        self.threshold_db: float = threshold_db
        self.hangover_frames: int = hangover_frames
        self._hangover_counter: int = 0

    # ------------------------------------------------------------------
    # VoiceActivityDetector interface
    # ------------------------------------------------------------------

    def process(self, chunk: np.ndarray) -> VADResult:
        """Classify *chunk* as speech or non-speech with hangover.

        Parameters
        ----------
        chunk:
            1-D float32 array, canonical 16 kHz format.

        Returns
        -------
        VADResult
            ``audio`` is the original chunk, **unmodified**.
            ``is_speech`` includes hangover hold-on.
            ``energy_db`` is the measured short-time energy.
        """
        if chunk.ndim != 1:
            raise ValueError(
                f"chunk must be 1-D, got shape {chunk.shape}"
            )

        energy_db = _rms_db(chunk)
        raw_speech = energy_db >= self.threshold_db

        if raw_speech:
            # Active speech: reload hangover counter.
            self._hangover_counter = self.hangover_frames
            is_speech = True
        elif self._hangover_counter > 0:
            # Hangover hold-on: below threshold but counter still active.
            self._hangover_counter -= 1
            is_speech = True
        else:
            is_speech = False

        return VADResult(audio=chunk, is_speech=is_speech, energy_db=energy_db)

    def reset(self) -> None:
        """Reset hangover counter.  Safe to call at any time."""
        self._hangover_counter = 0

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"EnergyVAD("
            f"threshold_db={self.threshold_db}, "
            f"hangover_frames={self.hangover_frames}, "
            f"_hangover_counter={self._hangover_counter})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rms_db(audio: np.ndarray) -> float:
    """Compute RMS energy of *audio* in dB relative to full scale (0 dB = 1.0).

    Returns ``_FLOOR_DB`` for silent/zero-length signals to avoid log(0).
    """
    if len(audio) == 0:
        return _FLOOR_DB
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if rms < 1e-12:
        return _FLOOR_DB
    return float(20.0 * np.log10(rms))
