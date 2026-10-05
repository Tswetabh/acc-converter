# Shared data classes (e.g. AudioChunk)
from dataclasses import dataclass
import numpy as np

@dataclass
class AudioChunk:
    data: np.ndarray
    sample_rate: int
