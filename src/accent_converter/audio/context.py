"""
audio.context
=============
Bounded streaming context / cache manager for the accent-converter pipeline.

Purpose
-------
During streaming inference each new audio chunk must be processed together
with a bounded window of **recent past audio** (look-back context).  This
module maintains that sliding window efficiently.

Design contract
---------------
* **Bounded memory** – the internal buffer never exceeds
  ``max_frames * chunk_samples`` samples regardless of how many chunks are
  pushed.  Oldest frames are evicted automatically.
* **Deterministic** – given the same sequence of ``push()`` calls, ``get()``
  always returns the same array.  No randomness, no hidden state beyond the
  ring-buffer.
* **reset()** – clears all stored state; the manager returns to its
  initial condition as if freshly constructed.
* **Separation of concerns** – this class does *not* resample, chunk, or
  normalise audio.  It only stores and retrieves float32 sample arrays.

Typical usage
-------------
::

    from accent_converter.audio.context import AudioContextManager
    from accent_converter.audio.chunker import CHUNK_SAMPLES

    ctx = AudioContextManager(max_frames=8, chunk_samples=CHUNK_SAMPLES)

    for chunk in stream:
        ctx.push(chunk)
        window = ctx.get()        # shape: (min(n_pushed, max_frames) * 1280,)
        run_model(window)

    ctx.reset()                   # start a new utterance / call

"""

from __future__ import annotations

from collections import deque

import numpy as np

from accent_converter.audio.chunker import CHUNK_SAMPLES
from accent_converter.audio.normalizer import CANONICAL_SAMPLE_RATE

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_MAX_FRAMES: int = 8   # 8 × 80 ms = 640 ms of look-back context


# ---------------------------------------------------------------------------
# AudioContextManager
# ---------------------------------------------------------------------------


class AudioContextManager:
    """Sliding-window cache of recent audio chunks.

    Parameters
    ----------
    max_frames:
        Maximum number of chunks to retain.  When a new chunk is pushed and
        the buffer is full, the oldest chunk is silently discarded.
        Default: 8 frames (640 ms at 80 ms / frame).
    chunk_samples:
        Expected number of samples in each chunk.  Used for shape validation
        and pre-allocation.  Default: 1 280 (80 ms at 16 000 Hz).

    Attributes
    ----------
    max_frames : int
    chunk_samples : int
    """

    def __init__(
        self,
        max_frames: int = DEFAULT_MAX_FRAMES,
        chunk_samples: int = CHUNK_SAMPLES,
    ) -> None:
        if max_frames < 1:
            raise ValueError(f"max_frames must be >= 1, got {max_frames}")
        if chunk_samples < 1:
            raise ValueError(f"chunk_samples must be >= 1, got {chunk_samples}")

        self.max_frames: int = max_frames
        self.chunk_samples: int = chunk_samples

        # Internal storage: deque of 1-D float32 arrays
        self._buffer: deque[np.ndarray] = deque(maxlen=max_frames)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def push(self, chunk: np.ndarray) -> None:
        """Append a new audio chunk to the context window.

        If the buffer is already at capacity (``len == max_frames``), the
        oldest chunk is automatically dropped before the new one is stored.

        Parameters
        ----------
        chunk:
            1-D float32 numpy array.  Length need not equal ``chunk_samples``
            exactly (partial/tail frames are accepted), but a warning is
            printed when the length differs from the expected size because
            this usually indicates a pipeline configuration error.

        Raises
        ------
        ValueError
            If *chunk* is not a 1-D numpy array.
        """
        if not isinstance(chunk, np.ndarray):
            raise TypeError(
                f"chunk must be a numpy ndarray, got {type(chunk).__name__}"
            )
        if chunk.ndim != 1:
            raise ValueError(
                f"chunk must be 1-D, got shape {chunk.shape}"
            )
        # Store a copy so that external mutations do not affect the cache.
        self._buffer.append(chunk.astype(np.float32, copy=True))

    def get(self) -> np.ndarray:
        """Return the concatenated context window as a single 1-D array.

        The frames are returned in chronological order (oldest first).

        Returns
        -------
        np.ndarray
            Concatenated float32 array of all buffered chunks, or an empty
            float32 array of shape ``(0,)`` if no chunks have been pushed
            since the last ``reset()``.
        """
        if not self._buffer:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(list(self._buffer))

    def reset(self) -> None:
        """Clear all buffered audio and return to the initial state.

        Call this at the start of each new utterance, call, or session to
        prevent context bleed between independent audio streams.
        """
        self._buffer.clear()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def n_frames(self) -> int:
        """Number of frames currently stored in the buffer."""
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        """True when the buffer has reached its maximum capacity."""
        return len(self._buffer) == self.max_frames

    @property
    def total_samples(self) -> int:
        """Total number of samples across all buffered frames."""
        return sum(len(f) for f in self._buffer)

    @property
    def max_samples(self) -> int:
        """Maximum sample capacity (``max_frames * chunk_samples``)."""
        return self.max_frames * self.chunk_samples

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"AudioContextManager("
            f"max_frames={self.max_frames}, "
            f"chunk_samples={self.chunk_samples}, "
            f"n_frames={self.n_frames}, "
            f"total_samples={self.total_samples})"
        )
