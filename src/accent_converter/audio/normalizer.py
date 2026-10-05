"""
audio.normalizer
================
Canonical audio normalisation for the accent-converter pipeline.

Canonical internal format
--------------------------
* Sample rate : 16 000 Hz  (enforced by the upstream Resampler)
* Channels    : mono (1-D numpy array, shape ``(N,)``)
* dtype       : float32
* Amplitude   : approximately [-1, 1]  (peak normalisation; silent frames
                are left unchanged to avoid amplifying silence artefacts)

This module is **stateless**.  Each function transforms an array and returns
a new array without modifying the input.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CANONICAL_SAMPLE_RATE: int = 16_000
CANONICAL_DTYPE = np.float32


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert a multi-channel audio array to mono by averaging channels.

    Parameters
    ----------
    audio:
        * 1-D array of shape ``(samples,)`` – returned as-is (no copy).
        * 2-D array of shape ``(channels, samples)`` – averaged over axis 0.
        * 2-D array of shape ``(samples, channels)`` – averaged over axis 1.

        The function infers the layout: if ``audio.shape[0] < audio.shape[1]``
        the first axis is assumed to be channels (channels-first layout).
        When the array is square or ambiguous it defaults to channels-first.

    Returns
    -------
    np.ndarray
        1-D mono array with the same dtype as the input.

    Raises
    ------
    ValueError
        If ``audio`` has more than 2 dimensions.
    """
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        # Decide layout heuristically:
        # channels-first  (C, T) → C is the short axis  → average axis 0
        # channels-last   (T, C) → T is the long axis   → average axis 1
        if audio.shape[0] <= audio.shape[1]:
            # channels-first (C, T)
            return audio.mean(axis=0).astype(audio.dtype)
        else:
            # channels-last (T, C)
            return audio.mean(axis=1).astype(audio.dtype)
    raise ValueError(
        f"audio must be 1-D or 2-D, got shape {audio.shape}"
    )


def to_float32(audio: np.ndarray) -> np.ndarray:
    """Cast an audio array to float32 with integer-to-float scaling.

    Integer dtypes are mapped to ``[-1, 1]`` using their full range:

    * int16  →  divided by 32 768
    * int32  →  divided by 2 147 483 648
    * uint8  →  ``(x - 128) / 128``  (centre the unsigned range)

    Floating-point dtypes (float16, float32, float64) are simply cast to
    float32 without rescaling.

    Parameters
    ----------
    audio:
        Input array of any numeric dtype.

    Returns
    -------
    np.ndarray
        float32 array.
    """
    if audio.dtype == np.float32:
        return audio

    if np.issubdtype(audio.dtype, np.floating):
        return audio.astype(np.float32)

    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32_768.0

    if audio.dtype == np.int32:
        return audio.astype(np.float32) / 2_147_483_648.0

    if audio.dtype == np.uint8:
        return (audio.astype(np.float32) - 128.0) / 128.0

    # Fallback: cast without scaling (covers int8, int64, etc.)
    return audio.astype(np.float32)


def peak_normalize(audio: np.ndarray, *, headroom: float = 1.0) -> np.ndarray:
    """Normalise amplitude so the absolute peak equals *headroom*.

    Silent frames (all zeros or numerically silent) are returned unchanged
    to avoid amplifying noise or silence artefacts.

    Parameters
    ----------
    audio:
        1-D float32 array of audio samples.
    headroom:
        Target peak amplitude.  Default is ``1.0`` (full scale).
        Use values slightly below 1 (e.g. ``0.99``) to add a small margin.

    Returns
    -------
    np.ndarray
        Peak-normalised float32 array.  Shape is identical to input.
    """
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        # Silent or near-silent: return a copy without rescaling.
        return audio.copy()
    return (audio / peak * headroom).astype(np.float32)


def normalize(
    audio: np.ndarray,
    *,
    peak_norm: bool = True,
    headroom: float = 1.0,
) -> np.ndarray:
    """Apply the full canonical normalisation chain.

    Steps (in order):

    1. ``to_mono``      – collapse channels if multi-channel
    2. ``to_float32``   – convert dtype and rescale integers
    3. ``peak_normalize`` – (optional) scale amplitude to *headroom*

    Parameters
    ----------
    audio:
        Input audio array.  May be any shape / dtype accepted by
        ``to_mono`` and ``to_float32``.
    peak_norm:
        If ``True`` (default), apply peak normalisation after dtype
        conversion.  Set to ``False`` when amplitude must be preserved
        (e.g. for level measurements).
    headroom:
        Target peak for normalisation.  Ignored when *peak_norm* is
        ``False``.

    Returns
    -------
    np.ndarray
        Canonical 1-D float32 array, amplitude in approximately ``[-1, 1]``.
    """
    audio = to_mono(audio)
    audio = to_float32(audio)
    if peak_norm:
        audio = peak_normalize(audio, headroom=headroom)
    return audio
