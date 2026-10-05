"""
audio.preprocessing
===================
Phase 3 audio preprocessing components.

Sub-packages
------------
vad/
    Voice activity detection.
    - EnergyVAD : energy-based VAD with hangover (pure numpy)

noise/
    Noise suppression.
    - NoiseReduceNoiseSuppressor : spectral gating via noisereduce library

aec/
    Acoustic echo cancellation.
    - PassthroughAEC : wires the pipeline slot; real AEC via browser WebRTC

Public API
----------
Import interfaces for type annotations::

    from accent_converter.audio.preprocessing import (
        VoiceActivityDetector, VADResult,
        NoiseSuppressor,
        AcousticEchoCanceller,
    )

Build backends from configuration::

    from accent_converter.audio.preprocessing import (
        build_vad, build_noise_suppressor, build_aec,
    )
    vad = build_vad({"backend": "energy", "threshold_db": -40})
"""

from accent_converter.audio.preprocessing.interfaces import (
    VoiceActivityDetector,
    NoiseSuppressor,
    AcousticEchoCanceller,
    VADResult,
)
from accent_converter.audio.preprocessing.factory import (
    build_vad,
    build_noise_suppressor,
    build_aec,
)

__all__ = [
    # Interfaces
    "VoiceActivityDetector",
    "NoiseSuppressor",
    "AcousticEchoCanceller",
    "VADResult",
    # Factory
    "build_vad",
    "build_noise_suppressor",
    "build_aec",
]
