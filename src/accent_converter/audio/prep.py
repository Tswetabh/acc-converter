"""
audio.prep  (Phase 3 redirect)
================================
This module is kept for backward compatibility.

All audio preprocessing logic has moved to:
    accent_converter.audio.preprocessing

Please import from there directly::

    from accent_converter.audio.preprocessing import (
        VoiceActivityDetector, build_vad,
        NoiseSuppressor, build_noise_suppressor,
        AcousticEchoCanceller, build_aec,
    )
"""

from accent_converter.audio.preprocessing import (  # noqa: F401
    VoiceActivityDetector,
    NoiseSuppressor,
    AcousticEchoCanceller,
    VADResult,
    build_vad,
    build_noise_suppressor,
    build_aec,
)
