"""
accent_converter.models
=======================
Model layer for the accent-converter pipeline.

Public API
----------
Import model **interfaces** from here for type annotations in pipeline code::

    from accent_converter.models import (
        ContentEncoder,
        SpeakerEncoder,
        AccentTranslator,
        SpeechSynthesizer,
    )

Import **factory functions** to instantiate backends from configuration::

    from accent_converter.models import (
        build_synthesizer,
        build_content_encoder,
        build_speaker_encoder,
        build_accent_translator,
    )

Never import concrete backend classes directly in pipeline code.
"""

from accent_converter.models.interfaces import (
    ContentEncoder,
    SpeakerEncoder,
    AccentTranslator,
    SpeechSynthesizer,
)
from accent_converter.models.factory import (
    build_synthesizer,
    build_content_encoder,
    build_speaker_encoder,
    build_accent_translator,
)

__all__ = [
    # Interfaces (for type annotations)
    "ContentEncoder",
    "SpeakerEncoder",
    "AccentTranslator",
    "SpeechSynthesizer",
    # Factory (for instantiation)
    "build_synthesizer",
    "build_content_encoder",
    "build_speaker_encoder",
    "build_accent_translator",
]
