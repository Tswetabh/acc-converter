"""
tests/unit/test_audio_context.py
==================================
Unit tests for accent_converter.audio.context.AudioContextManager

Covers:
- cache size limit (bounded memory)
- oldest frame evicted when full
- reset() clears all state
- get() returns chronological concatenation
- get() on empty manager returns empty array
- n_frames and is_full properties
- total_samples accounting
- push() type / shape validation
- deterministic behaviour (same pushes → same get())
- partial-length chunks accepted
"""

import numpy as np
import pytest

from accent_converter.audio.context import AudioContextManager, DEFAULT_MAX_FRAMES
from accent_converter.audio.chunker import CHUNK_SAMPLES


# ===========================================================================
# Helpers
# ===========================================================================


def _chunk(value: float, n: int = CHUNK_SAMPLES, dtype=np.float32) -> np.ndarray:
    """Return a constant-value 1-D float32 chunk for easy content verification."""
    return np.full(n, value, dtype=dtype)


# ===========================================================================
# Construction
# ===========================================================================


class TestConstruction:
    def test_default_parameters(self):
        ctx = AudioContextManager()
        assert ctx.max_frames == DEFAULT_MAX_FRAMES
        assert ctx.chunk_samples == CHUNK_SAMPLES

    def test_custom_parameters(self):
        ctx = AudioContextManager(max_frames=4, chunk_samples=512)
        assert ctx.max_frames == 4
        assert ctx.chunk_samples == 512

    def test_invalid_max_frames(self):
        with pytest.raises(ValueError, match="max_frames"):
            AudioContextManager(max_frames=0)

    def test_invalid_chunk_samples(self):
        with pytest.raises(ValueError, match="chunk_samples"):
            AudioContextManager(chunk_samples=0)


# ===========================================================================
# Empty state
# ===========================================================================


class TestEmptyState:
    def test_get_empty_returns_empty_array(self):
        ctx = AudioContextManager(max_frames=4)
        result = ctx.get()
        assert result.dtype == np.float32
        assert result.shape == (0,)

    def test_n_frames_initially_zero(self):
        ctx = AudioContextManager()
        assert ctx.n_frames == 0

    def test_is_full_initially_false(self):
        ctx = AudioContextManager(max_frames=4)
        assert not ctx.is_full

    def test_total_samples_initially_zero(self):
        ctx = AudioContextManager()
        assert ctx.total_samples == 0


# ===========================================================================
# Push and get
# ===========================================================================


class TestPushGet:
    def test_single_push(self):
        ctx = AudioContextManager(max_frames=4)
        chunk = _chunk(1.0)
        ctx.push(chunk)
        result = ctx.get()
        assert result.shape == (CHUNK_SAMPLES,)
        np.testing.assert_array_equal(result, chunk)

    def test_multiple_pushes_concatenated_chronologically(self):
        """get() returns oldest-first concatenation."""
        ctx = AudioContextManager(max_frames=4)
        ctx.push(_chunk(1.0))
        ctx.push(_chunk(2.0))
        ctx.push(_chunk(3.0))
        result = ctx.get()
        expected = np.concatenate([_chunk(1.0), _chunk(2.0), _chunk(3.0)])
        np.testing.assert_array_equal(result, expected)

    def test_n_frames_increments(self):
        ctx = AudioContextManager(max_frames=4)
        for i in range(3):
            ctx.push(_chunk(float(i)))
            assert ctx.n_frames == i + 1

    def test_total_samples_increments(self):
        ctx = AudioContextManager(max_frames=4)
        for i in range(3):
            ctx.push(_chunk(float(i)))
        assert ctx.total_samples == 3 * CHUNK_SAMPLES

    def test_push_stores_copy(self):
        """Mutating the original chunk after push must not affect cache."""
        ctx = AudioContextManager(max_frames=4)
        chunk = _chunk(1.0)
        ctx.push(chunk)
        chunk[:] = 99.0   # mutate original
        result = ctx.get()
        np.testing.assert_array_equal(result, _chunk(1.0))


# ===========================================================================
# Cache size limit (bounded memory)
# ===========================================================================


class TestCacheSizeLimit:
    def test_exceeding_max_frames_evicts_oldest(self):
        """When buffer is full, the oldest frame is evicted on push."""
        max_frames = 3
        ctx = AudioContextManager(max_frames=max_frames)
        # Push max_frames + 1 chunks
        for i in range(max_frames + 1):
            ctx.push(_chunk(float(i)))
        # Buffer should hold only the last max_frames chunks
        assert ctx.n_frames == max_frames
        result = ctx.get()
        # Frame 0 (value=0.0) should be gone; frames 1, 2, 3 remain
        expected = np.concatenate([_chunk(1.0), _chunk(2.0), _chunk(3.0)])
        np.testing.assert_array_equal(result, expected)

    def test_is_full_when_at_capacity(self):
        max_frames = 4
        ctx = AudioContextManager(max_frames=max_frames)
        for i in range(max_frames):
            ctx.push(_chunk(float(i)))
        assert ctx.is_full

    def test_n_frames_never_exceeds_max(self):
        max_frames = 5
        ctx = AudioContextManager(max_frames=max_frames)
        for i in range(max_frames * 3):   # push 3× capacity
            ctx.push(_chunk(float(i)))
        assert ctx.n_frames == max_frames

    def test_buffer_bounded_after_many_pushes(self):
        """Buffer size stays constant after many pushes beyond capacity."""
        ctx = AudioContextManager(max_frames=4)
        for i in range(100):
            ctx.push(_chunk(float(i)))
        assert ctx.n_frames == 4
        assert len(ctx.get()) == 4 * CHUNK_SAMPLES

    def test_max_samples_property(self):
        ctx = AudioContextManager(max_frames=8, chunk_samples=1_280)
        assert ctx.max_samples == 8 * 1_280


# ===========================================================================
# reset()
# ===========================================================================


class TestReset:
    def test_reset_clears_buffer(self):
        ctx = AudioContextManager(max_frames=4)
        for _ in range(4):
            ctx.push(_chunk(1.0))
        ctx.reset()
        assert ctx.n_frames == 0
        assert len(ctx.get()) == 0

    def test_reset_returns_to_initial_state(self):
        """After reset(), manager behaves as if freshly constructed."""
        ctx = AudioContextManager(max_frames=4)
        for _ in range(4):
            ctx.push(_chunk(1.0))
        ctx.reset()
        # Push a single new frame
        ctx.push(_chunk(2.0))
        assert ctx.n_frames == 1
        np.testing.assert_array_equal(ctx.get(), _chunk(2.0))

    def test_reset_on_empty_is_safe(self):
        """reset() on an already-empty manager should not raise."""
        ctx = AudioContextManager(max_frames=4)
        ctx.reset()   # must not raise
        assert ctx.n_frames == 0

    def test_double_reset_is_safe(self):
        ctx = AudioContextManager(max_frames=4)
        for _ in range(3):
            ctx.push(_chunk(1.0))
        ctx.reset()
        ctx.reset()   # second reset must not raise
        assert ctx.n_frames == 0


# ===========================================================================
# Determinism
# ===========================================================================


class TestDeterminism:
    def test_same_pushes_same_output(self):
        """Two managers receiving the same chunks must produce identical get()."""
        ctx_a = AudioContextManager(max_frames=4)
        ctx_b = AudioContextManager(max_frames=4)
        chunks = [_chunk(float(i)) for i in range(3)]
        for c in chunks:
            ctx_a.push(c)
            ctx_b.push(c)
        np.testing.assert_array_equal(ctx_a.get(), ctx_b.get())

    def test_get_is_idempotent(self):
        """Calling get() twice without pushing returns the same result."""
        ctx = AudioContextManager(max_frames=4)
        ctx.push(_chunk(1.0))
        ctx.push(_chunk(2.0))
        np.testing.assert_array_equal(ctx.get(), ctx.get())


# ===========================================================================
# Input validation
# ===========================================================================


class TestInputValidation:
    def test_push_non_ndarray_raises(self):
        ctx = AudioContextManager()
        with pytest.raises(TypeError, match="ndarray"):
            ctx.push([1.0, 2.0, 3.0])  # type: ignore[arg-type]

    def test_push_2d_raises(self):
        ctx = AudioContextManager()
        with pytest.raises(ValueError, match="1-D"):
            ctx.push(np.zeros((2, CHUNK_SAMPLES), dtype=np.float32))


# ===========================================================================
# Partial-length chunks
# ===========================================================================


class TestPartialChunks:
    def test_partial_chunk_accepted(self):
        """Chunks shorter than chunk_samples are stored without error."""
        ctx = AudioContextManager(max_frames=4)
        short = _chunk(1.0, n=500)
        ctx.push(short)
        result = ctx.get()
        assert len(result) == 500

    def test_mixed_length_chunks(self):
        """Full and partial chunks can coexist in the buffer."""
        ctx = AudioContextManager(max_frames=4)
        ctx.push(_chunk(1.0, n=CHUNK_SAMPLES))
        ctx.push(_chunk(2.0, n=500))
        result = ctx.get()
        assert len(result) == CHUNK_SAMPLES + 500
