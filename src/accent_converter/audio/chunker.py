"""
audio.chunker
=============
80 ms frame chunker for the accent-converter audio pipeline.

At the canonical sample rate of 16 000 Hz, an 80 ms chunk is exactly
1 280 samples.

Design contract
---------------
* The chunker is **stateless** — it does not accumulate a ring-buffer
  internally.  The caller owns the accumulation buffer.
* ``chunk_audio`` operates on an already-normalised 1-D float32 array
  and returns a list of fixed-size frames.
* Partial final frames (tail) are handled explicitly:

  - ``tail='drop'``    (default) – discard samples that do not fill a
    complete frame.
  - ``tail='keep'``   – return the partial frame as-is (shorter than
    chunk_size).
  - ``tail='pad'``    – zero-pad the partial frame to chunk_size.

* ``frame_generator`` is a generator variant that yields one frame at a
  time, which avoids allocating the full list when processing long streams.

Constants
---------
CHUNK_MS        : int   – nominal chunk duration in milliseconds (80)
CHUNK_SAMPLES   : int   – samples per chunk at 16 000 Hz (1 280)
"""

from __future__ import annotations

from typing import Generator, Literal

import numpy as np

from accent_converter.audio.normalizer import CANONICAL_SAMPLE_RATE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHUNK_MS: int = 80
CHUNK_SAMPLES: int = int(CANONICAL_SAMPLE_RATE * CHUNK_MS / 1000)  # 1 280


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_audio(
    audio: np.ndarray,
    chunk_size_ms: int = CHUNK_MS,
    sample_rate: int = CANONICAL_SAMPLE_RATE,
    tail: Literal["drop", "keep", "pad"] = "drop",
) -> list[np.ndarray]:
    """Split *audio* into fixed-size chunks of *chunk_size_ms* milliseconds.

    Parameters
    ----------
    audio:
        1-D float32 numpy array of audio samples.
    chunk_size_ms:
        Duration of each chunk in milliseconds.  Default: 80 ms.
    sample_rate:
        Sample rate of *audio* in Hz.  Default: 16 000 Hz (canonical).
    tail:
        How to handle the final partial chunk when ``len(audio)`` is not an
        exact multiple of ``chunk_size``.

        * ``'drop'`` – discard leftover samples (default).
        * ``'keep'`` – return the partial chunk with its actual (shorter)
          length.
        * ``'pad'``  – zero-pad the partial chunk to ``chunk_size``.

    Returns
    -------
    list[np.ndarray]
        List of 1-D arrays.  Each complete chunk has exactly ``chunk_size``
        samples.  The final chunk may be shorter if ``tail='keep'``.

    Raises
    ------
    ValueError
        If *audio* is not 1-D, or if *tail* is not one of the accepted
        literals.
    """
    if audio.ndim != 1:
        raise ValueError(
            f"audio must be 1-D, got shape {audio.shape}. "
            "Run normalizer.to_mono() first."
        )
    valid_tails = ("drop", "keep", "pad")
    if tail not in valid_tails:
        raise ValueError(f"tail must be one of {valid_tails}, got {tail!r}")

    chunk_size = _chunk_size_samples(chunk_size_ms, sample_rate)
    return list(frame_generator(audio, chunk_size=chunk_size, tail=tail))


def frame_generator(
    audio: np.ndarray,
    chunk_size: int = CHUNK_SAMPLES,
    tail: Literal["drop", "keep", "pad"] = "drop",
) -> Generator[np.ndarray, None, None]:
    """Yield fixed-size frames from *audio* one at a time.

    This is the memory-efficient generator variant of ``chunk_audio``.
    Prefer this for long audio streams.

    Parameters
    ----------
    audio:
        1-D numpy array.
    chunk_size:
        Number of samples per frame.  Default: 1 280 (80 ms at 16 kHz).
    tail:
        Handling of the final partial frame.  Same semantics as
        ``chunk_audio``.

    Yields
    ------
    np.ndarray
        Successive non-overlapping frames of length ``chunk_size``
        (or shorter for the tail when ``tail='keep'``).
    """
    n = len(audio)
    start = 0
    while start + chunk_size <= n:
        yield audio[start : start + chunk_size]
        start += chunk_size

    # Handle tail
    remainder = n - start
    if remainder > 0:
        if tail == "keep":
            yield audio[start:]
        elif tail == "pad":
            frame = np.zeros(chunk_size, dtype=audio.dtype)
            frame[:remainder] = audio[start:]
            yield frame
        # tail == "drop": do nothing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk_size_samples(chunk_size_ms: int, sample_rate: int) -> int:
    """Convert chunk duration in ms to sample count.

    Uses integer arithmetic to avoid floating-point rounding surprises.
    Raises ``ValueError`` if the result is not an exact integer.
    """
    numerator = sample_rate * chunk_size_ms
    if numerator % 1000 != 0:
        raise ValueError(
            f"chunk_size_ms={chunk_size_ms} ms at sample_rate={sample_rate} Hz "
            f"does not produce an integer sample count "
            f"({numerator / 1000:.4f} samples)."
        )
    return numerator // 1000
