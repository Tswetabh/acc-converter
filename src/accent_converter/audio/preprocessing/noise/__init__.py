# audio/preprocessing/noise/__init__.py
from accent_converter.audio.preprocessing.noise.noisereduce_backend import (
    NoiseReduceNoiseSuppressor,
)

__all__ = ["NoiseReduceNoiseSuppressor"]
