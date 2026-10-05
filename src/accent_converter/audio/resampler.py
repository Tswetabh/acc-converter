"""
audio.resampler
===============
Sample-rate conversion for the accent-converter pipeline.

Converts audio from common input sample rates to the canonical 16 000 Hz
internal format using polyphase rational resampling (``scipy.signal.resample_poly``).

Supported common input rates
-----------------------------
* 8 000 Hz
* 11 025 Hz
* 16 000 Hz  (passthrough – no computation)
* 22 050 Hz
* 32 000 Hz
* 44 100 Hz
* 48 000 Hz
* 96 000 Hz

Any positive integer sample rate is accepted; the table above merely lists
rates that have been tested and have known GCD ratios pre-computed.

Design notes
------------
* ``resample_poly`` performs exact rational resampling (up/down by integer
  factors derived from GCD reduction), which avoids the aliasing artefacts
  of FFT-based resampling and the fractional-delay errors of linear
  interpolation.
* The anti-aliasing filter built into ``resample_poly`` is applied
  automatically.
* The resampler is **stateless**; each call is independent.
* dtype is preserved.  Convert to float32 separately via ``normalizer.to_float32``.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import resample_poly

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SAMPLE_RATE: int = 16_000

# Pre-computed (up, down) integer factors for the most common input rates.
# Stored so that math.gcd is called once at module import, not per call.
_RATIO_CACHE: dict[int, tuple[int, int]] = {}


def _get_ratio(orig_sr: int, target_sr: int) -> tuple[int, int]:
    """Return (up, down) integer factors for resample_poly.

    Reduces the fraction ``target_sr / orig_sr`` by the GCD so that
    ``resample_poly`` uses the smallest possible filter.
    """
    key = (orig_sr, target_sr)
    if key not in _RATIO_CACHE:
        g = math.gcd(target_sr, orig_sr)
        _RATIO_CACHE[key] = (target_sr // g, orig_sr // g)
    return _RATIO_CACHE[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resample(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    """Resample *audio* from *orig_sr* to *target_sr* using polyphase filtering.

    Parameters
    ----------
    audio:
        1-D numpy array of audio samples.  Must already be mono; for
        multi-channel audio call ``normalizer.to_mono`` first.
    orig_sr:
        Original sample rate in Hz (positive integer).
    target_sr:
        Target sample rate in Hz.  Defaults to 16 000 Hz.

    Returns
    -------
    np.ndarray
        Resampled array with length ``round(len(audio) * target_sr / orig_sr)``.
        dtype is float64 (scipy output); call ``normalizer.to_float32`` to
        convert.

    Raises
    ------
    ValueError
        If *orig_sr* or *target_sr* are not positive integers, or if
        *audio* is not 1-D.
    """
    if not isinstance(orig_sr, int) or orig_sr <= 0:
        raise ValueError(f"orig_sr must be a positive integer, got {orig_sr!r}")
    if not isinstance(target_sr, int) or target_sr <= 0:
        raise ValueError(f"target_sr must be a positive integer, got {target_sr!r}")
    if audio.ndim != 1:
        raise ValueError(
            f"audio must be 1-D (mono). Got shape {audio.shape}. "
            "Call normalizer.to_mono() first."
        )

    if orig_sr == target_sr:
        # Passthrough: no computation, return a copy to preserve immutability.
        return audio.copy()

    up, down = _get_ratio(orig_sr, target_sr)
    return resample_poly(audio, up, down).astype(audio.dtype)
