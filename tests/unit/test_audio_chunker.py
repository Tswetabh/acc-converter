"""
tests/unit/test_audio_chunker.py
==================================
Unit tests for accent_converter.audio.chunker

Covers:
- exact 80 ms chunk size at 16 kHz (1 280 samples)
- multiple sequential full chunks
- partial (tail) chunk with tail='keep'
- partial chunk is dropped with tail='drop' (default)
- partial chunk is padded with tail='pad'
- exact-multiple audio has no tail
- empty audio returns empty list
- CHUNK_SAMPLES and CHUNK_MS constants
- 2-D input raises ValueError
- invalid tail raises ValueError
- frame_generator yields same results as chunk_audio
"""

import numpy as np
import pytest

from accent_converter.audio.chunker import (
    chunk_audio,
    frame_generator,
    CHUNK_SAMPLES,
    CHUNK_MS,
    _chunk_size_samples,
)
from accent_converter.audio.normalizer import CANONICAL_SAMPLE_RATE


# ===========================================================================
# Helpers
# ===========================================================================


def _audio(n_samples: int, dtype=np.float32) -> np.ndarray:
    """Ramp audio: values 0, 1, 2, ..., n_samples-1 (distinguishable frames)."""
    return np.arange(n_samples, dtype=dtype)


# ===========================================================================
# Constants
# ===========================================================================


class TestConstants:
    def test_chunk_ms_is_80(self):
        assert CHUNK_MS == 80

    def test_chunk_samples_is_1280(self):
        """At 16 kHz, 80 ms = 1 280 samples."""
        assert CHUNK_SAMPLES == 1_280
        assert CHUNK_SAMPLES == int(CANONICAL_SAMPLE_RATE * CHUNK_MS / 1000)


# ===========================================================================
# Exact chunk size
# ===========================================================================


class TestExactChunkSize:
    def test_single_chunk_exact_length(self):
        """Exactly one 80 ms chunk → list of length 1, chunk has 1 280 samples."""
        audio = _audio(CHUNK_SAMPLES)
        chunks = chunk_audio(audio)
        assert len(chunks) == 1
        assert len(chunks[0]) == CHUNK_SAMPLES

    def test_single_chunk_content(self):
        """Chunk content matches the original samples."""
        audio = _audio(CHUNK_SAMPLES)
        chunks = chunk_audio(audio)
        np.testing.assert_array_equal(chunks[0], audio)


# ===========================================================================
# Multiple sequential chunks
# ===========================================================================


class TestMultipleChunks:
    def test_three_full_chunks(self):
        """3 × CHUNK_SAMPLES audio → 3 chunks, each 1 280 samples."""
        audio = _audio(3 * CHUNK_SAMPLES)
        chunks = chunk_audio(audio)
        assert len(chunks) == 3
        for i, chunk in enumerate(chunks):
            assert len(chunk) == CHUNK_SAMPLES
            expected = audio[i * CHUNK_SAMPLES: (i + 1) * CHUNK_SAMPLES]
            np.testing.assert_array_equal(chunk, expected)

    def test_ten_chunks(self):
        audio = _audio(10 * CHUNK_SAMPLES)
        chunks = chunk_audio(audio)
        assert len(chunks) == 10
        for chunk in chunks:
            assert len(chunk) == CHUNK_SAMPLES

    def test_chunks_are_non_overlapping_and_contiguous(self):
        """Concatenated chunks exactly reconstruct the original audio."""
        audio = _audio(5 * CHUNK_SAMPLES)
        chunks = chunk_audio(audio)
        reconstructed = np.concatenate(chunks)
        np.testing.assert_array_equal(reconstructed, audio)


# ===========================================================================
# Partial / tail chunks
# ===========================================================================


class TestPartialChunks:
    EXTRA = 100   # samples beyond full chunks

    def test_tail_drop_default(self):
        """Default tail='drop': leftover samples are discarded."""
        audio = _audio(2 * CHUNK_SAMPLES + self.EXTRA)
        chunks = chunk_audio(audio, tail="drop")
        assert len(chunks) == 2
        assert all(len(c) == CHUNK_SAMPLES for c in chunks)

    def test_tail_drop_explicit(self):
        audio = _audio(CHUNK_SAMPLES + self.EXTRA)
        chunks = chunk_audio(audio, tail="drop")
        assert len(chunks) == 1

    def test_tail_keep(self):
        """tail='keep': partial chunk is returned with its actual length."""
        audio = _audio(CHUNK_SAMPLES + self.EXTRA)
        chunks = chunk_audio(audio, tail="keep")
        assert len(chunks) == 2
        assert len(chunks[0]) == CHUNK_SAMPLES
        assert len(chunks[1]) == self.EXTRA
        np.testing.assert_array_equal(chunks[1], audio[CHUNK_SAMPLES:])

    def test_tail_pad(self):
        """tail='pad': partial chunk is zero-padded to CHUNK_SAMPLES."""
        audio = _audio(CHUNK_SAMPLES + self.EXTRA)
        chunks = chunk_audio(audio, tail="pad")
        assert len(chunks) == 2
        assert len(chunks[1]) == CHUNK_SAMPLES
        # First EXTRA samples match
        np.testing.assert_array_equal(chunks[1][: self.EXTRA], audio[CHUNK_SAMPLES:])
        # Remaining samples are zero
        np.testing.assert_array_equal(chunks[1][self.EXTRA:], np.zeros(CHUNK_SAMPLES - self.EXTRA, dtype=audio.dtype))

    def test_no_tail_exact_multiple(self):
        """Audio that is an exact multiple has no tail regardless of mode."""
        audio = _audio(4 * CHUNK_SAMPLES)
        for tail in ("drop", "keep", "pad"):
            chunks = chunk_audio(audio, tail=tail)
            assert len(chunks) == 4


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_empty_audio_drop(self):
        audio = _audio(0)
        assert chunk_audio(audio, tail="drop") == []

    def test_empty_audio_keep(self):
        audio = _audio(0)
        assert chunk_audio(audio, tail="keep") == []

    def test_audio_shorter_than_chunk_drop(self):
        """Audio shorter than one chunk with tail='drop' → empty list."""
        audio = _audio(CHUNK_SAMPLES - 1)
        chunks = chunk_audio(audio, tail="drop")
        assert chunks == []

    def test_audio_shorter_than_chunk_keep(self):
        """Audio shorter than one chunk with tail='keep' → one partial chunk."""
        audio = _audio(CHUNK_SAMPLES - 1)
        chunks = chunk_audio(audio, tail="keep")
        assert len(chunks) == 1
        assert len(chunks[0]) == CHUNK_SAMPLES - 1


# ===========================================================================
# Error handling
# ===========================================================================


class TestErrors:
    def test_2d_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            chunk_audio(np.zeros((2, 1000), dtype=np.float32))

    def test_invalid_tail_raises(self):
        with pytest.raises(ValueError, match="tail"):
            chunk_audio(_audio(CHUNK_SAMPLES), tail="invalid")  # type: ignore[arg-type]

    def test_non_integer_chunk_ms_raises(self):
        """chunk_size_ms that produces a non-integer sample count raises.

        Example: 33 ms at 44 100 Hz → 44100 * 33 / 1000 = 1455.3 samples (non-integer).
        Note: 33 ms at 16 000 Hz → 528.0 samples, which IS an integer and does NOT raise.
        """
        with pytest.raises(ValueError, match="integer sample count"):
            _chunk_size_samples(chunk_size_ms=33, sample_rate=44_100)


# ===========================================================================
# frame_generator
# ===========================================================================


class TestFrameGenerator:
    def test_generator_matches_chunk_audio(self):
        """frame_generator must produce identical results to chunk_audio."""
        audio = _audio(3 * CHUNK_SAMPLES + 500)
        list_result = chunk_audio(audio, tail="keep")
        gen_result = list(frame_generator(audio, tail="keep"))
        assert len(list_result) == len(gen_result)
        for a, b in zip(list_result, gen_result):
            np.testing.assert_array_equal(a, b)

    def test_generator_is_lazy(self):
        """frame_generator is a generator (not a list)."""
        import types
        audio = _audio(CHUNK_SAMPLES * 5)
        result = frame_generator(audio)
        assert isinstance(result, types.GeneratorType)

    def test_custom_chunk_size(self):
        """Non-default chunk_size parameter is respected."""
        audio = _audio(600)
        chunks = list(frame_generator(audio, chunk_size=200, tail="drop"))
        assert len(chunks) == 3
        for c in chunks:
            assert len(c) == 200
