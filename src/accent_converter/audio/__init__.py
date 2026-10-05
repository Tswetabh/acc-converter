"""
accent_converter.audio
======================
Audio foundation for the accent-converter pipeline.

Sub-modules
-----------
normalizer
    Canonical audio normalisation:
    - to_mono()         – collapse channels
    - to_float32()      – dtype conversion with integer scaling
    - peak_normalize()  – amplitude normalisation to [-1, 1]
    - normalize()       – combined pipeline convenience function

resampler
    Sample-rate conversion:
    - resample()        – polyphase rational resampling via scipy

chunker
    Fixed-size frame extraction:
    - chunk_audio()     – split array into list of frames
    - frame_generator() – generator variant for memory efficiency
    - CHUNK_MS          – nominal 80 ms chunk duration constant
    - CHUNK_SAMPLES     – 1 280 samples per chunk at 16 kHz constant

context
    Bounded sliding-window cache:
    - AudioContextManager  – push/get/reset interface with bounded memory
"""

from accent_converter.audio.normalizer import (
    CANONICAL_SAMPLE_RATE,
    CANONICAL_DTYPE,
    to_mono,
    to_float32,
    peak_normalize,
    normalize,
)
from accent_converter.audio.resampler import resample, TARGET_SAMPLE_RATE
from accent_converter.audio.chunker import (
    chunk_audio,
    frame_generator,
    CHUNK_MS,
    CHUNK_SAMPLES,
)
from accent_converter.audio.context import AudioContextManager

__all__ = [
    # normalizer
    "CANONICAL_SAMPLE_RATE",
    "CANONICAL_DTYPE",
    "to_mono",
    "to_float32",
    "peak_normalize",
    "normalize",
    # resampler
    "resample",
    "TARGET_SAMPLE_RATE",
    # chunker
    "chunk_audio",
    "frame_generator",
    "CHUNK_MS",
    "CHUNK_SAMPLES",
    # context
    "AudioContextManager",
]
