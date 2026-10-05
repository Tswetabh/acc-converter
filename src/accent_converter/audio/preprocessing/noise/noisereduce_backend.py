"""
audio.preprocessing.noise.noisereduce_backend
===============================================
Noise suppressor backend using the ``noisereduce`` library.

Algorithm (noisereduce)
-----------------------
Spectral gating — estimates the noise floor from the signal statistics (or an
explicit noise clip) and suppresses frequency bins below the estimated noise
threshold.

Two modes
---------
* **Stationary** (``stationary=True``): assumes the noise profile is constant.
  Fast, low-latency.  Good for fan / air-conditioning / microphone hiss.
* **Non-stationary** (``stationary=False``, default): adaptive noise tracking.
  Better for keyboard / background chatter.  Higher CPU.

Important limitations
---------------------
* ``noisereduce.reduce_noise`` operates on a **full buffer** per call — it is
  not frame-synchronous.  Calling it on each 80 ms chunk independently works
  but may produce boundary artefacts.  For production streaming, accumulate
  a longer segment (≥ 500 ms) before suppressing.
* No persistent noise model is maintained across calls in this implementation.
  ``reset()`` is therefore a no-op.
* The optional ``y_noise`` parameter can be used to provide an explicit
  noise-only clip for better suppression during a known-quiet period.

Backend name: ``"noisereduce"``

Configuration
-------------
::

    preprocessing:
      noise_suppressor:
        backend: noisereduce
        stationary: false      # true = faster, false = adaptive (default)
        prop_decrease: 1.0     # proportion of noise to remove (0.0–1.0)
"""

from __future__ import annotations

import numpy as np

from accent_converter.audio.preprocessing.interfaces import NoiseSuppressor
from accent_converter.audio.normalizer import CANONICAL_SAMPLE_RATE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_STATIONARY: bool = False
DEFAULT_PROP_DECREASE: float = 1.0


class NoiseReduceNoiseSuppressor(NoiseSuppressor):
    """Spectral-gating noise suppressor backed by ``noisereduce``.

    Parameters
    ----------
    sample_rate:
        Sample rate of the audio in Hz.  Must match the pipeline canonical
        rate (16 000 Hz).
    stationary:
        If ``True``, use stationary noise estimation (faster).
        If ``False`` (default), use non-stationary adaptive estimation.
    prop_decrease:
        Proportion of noise reduction applied.  ``1.0`` = full reduction,
        ``0.5`` = half.  Default: ``1.0``.

    Attributes
    ----------
    sample_rate : int
    stationary : bool
    prop_decrease : float
    """

    BACKEND_NAME: str = "noisereduce"

    def __init__(
        self,
        sample_rate: int = CANONICAL_SAMPLE_RATE,
        stationary: bool = DEFAULT_STATIONARY,
        prop_decrease: float = DEFAULT_PROP_DECREASE,
    ) -> None:
        self.sample_rate: int = sample_rate
        self.stationary: bool = stationary
        self.prop_decrease: float = prop_decrease

    # ------------------------------------------------------------------
    # NoiseSuppressor interface
    # ------------------------------------------------------------------

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply spectral-gating noise suppression to *audio*.

        Parameters
        ----------
        audio:
            1-D float32 array, canonical 16 kHz format.

        Returns
        -------
        np.ndarray
            Cleaned float32 array, same shape as input.

        Notes
        -----
        Very short arrays (< 256 samples) are returned unchanged because
        ``noisereduce`` needs at least one FFT window to function.
        """
        if audio.ndim != 1:
            raise ValueError(f"audio must be 1-D, got shape {audio.shape}")
        if len(audio) < 256:
            # Too short for spectral estimation — return unchanged.
            return audio.copy()

        import noisereduce as nr  # deferred import — not required if unused

        cleaned = nr.reduce_noise(
            y=audio.astype(np.float32),
            sr=self.sample_rate,
            stationary=self.stationary,
            prop_decrease=self.prop_decrease,
            use_tqdm=False,
        )
        return cleaned.astype(np.float32)

    def reset(self) -> None:
        """No-op — noisereduce does not maintain persistent state across calls."""
        pass

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"NoiseReduceNoiseSuppressor("
            f"stationary={self.stationary}, "
            f"prop_decrease={self.prop_decrease})"
        )
