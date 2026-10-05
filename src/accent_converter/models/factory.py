"""
models.factory
==============
Factory functions that read a configuration dict and return the appropriate
model backend instances.

The pipeline must obtain **all** model instances from this module.  It must
never import concrete backend classes directly.

Usage
-----
::

    from accent_converter.models.factory import (
        build_synthesizer,
        build_content_encoder,
        build_speaker_encoder,
        build_accent_translator,
    )

    synthesizer = build_synthesizer(config["synthesis"])
    # synthesizer is a SpeechSynthesizer — the pipeline does not know
    # whether it is a VocoderSynthesizer or TVTSynSynthesizer.

Configuration keys
------------------
synthesis:
  backend: vocoder | tvtsyn
  checkpoint: <path>          # optional — passed to load() if present

models:
  content_encoder:
    backend: hubert
    checkpoint: <path>
  speaker_encoder:
    backend: ecapa
    checkpoint: <path>
  accent_translator:
    backend: naive
    checkpoint: <path>
    target_accent: us | uk | neutral

Adding a new backend
--------------------
1. Create a new module under ``models/backends/<role>/<name>.py``.
2. Subclass the appropriate interface from ``models.interfaces``.
3. Register the new backend in the ``_SYNTHESIS_BACKENDS``, ``_CONTENT_BACKENDS``,
   ``_SPEAKER_BACKENDS``, or ``_ACCENT_BACKENDS`` dict below.
4. The pipeline requires no changes.
"""

from __future__ import annotations

from typing import Any

from accent_converter.models.interfaces import (
    ContentEncoder,
    SpeakerEncoder,
    AccentTranslator,
    SpeechSynthesizer,
)

# ---------------------------------------------------------------------------
# Backend registries
# ---------------------------------------------------------------------------
# Map backend name (str) → class.  Add new backends here only.

def _synthesis_backends() -> dict[str, type[SpeechSynthesizer]]:
    from accent_converter.models.backends.synthesis.vocoder import VocoderSynthesizer
    from accent_converter.models.backends.synthesis.tvtsyn import TVTSynSynthesizer
    return {
        VocoderSynthesizer.BACKEND_NAME: VocoderSynthesizer,
        TVTSynSynthesizer.BACKEND_NAME: TVTSynSynthesizer,
    }


def _content_backends() -> dict[str, type[ContentEncoder]]:
    from accent_converter.models.backends.content.hubert import HubertContentEncoder
    return {
        HubertContentEncoder.BACKEND_NAME: HubertContentEncoder,
    }


def _speaker_backends() -> dict[str, type[SpeakerEncoder]]:
    from accent_converter.models.backends.speaker.ecapa import EcapaSpeakerEncoder
    return {
        EcapaSpeakerEncoder.BACKEND_NAME: EcapaSpeakerEncoder,
    }


def _accent_backends() -> dict[str, type[AccentTranslator]]:
    from accent_converter.models.backends.accent.naive import NaiveAccentTranslator
    from accent_converter.models.backends.accent.conformer import ConformerAccentTranslator
    return {
        NaiveAccentTranslator.BACKEND_NAME: NaiveAccentTranslator,
        ConformerAccentTranslator.BACKEND_NAME: ConformerAccentTranslator,
    }


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def build_synthesizer(config: dict[str, Any]) -> SpeechSynthesizer:
    """Instantiate a :class:`~accent_converter.models.interfaces.SpeechSynthesizer`.

    Parameters
    ----------
    config:
        The ``synthesis`` sub-dict from the project configuration, e.g.::

            {"backend": "vocoder", "checkpoint": "path/to/vocoder.ckpt"}

    Returns
    -------
    SpeechSynthesizer
        An instance of the requested backend.  The caller should call
        ``synthesizer.load(checkpoint)`` separately if a checkpoint is
        specified.

    Raises
    ------
    KeyError
        If ``config["backend"]`` is not registered.
    ValueError
        If ``config`` does not contain a ``"backend"`` key.
    """
    backend_name = _require_backend_key(config)
    backends = _synthesis_backends()
    cls = _lookup(backend_name, backends, role="synthesis")
    return cls()


def build_content_encoder(config: dict[str, Any]) -> ContentEncoder:
    """Instantiate a :class:`~accent_converter.models.interfaces.ContentEncoder`."""
    backend_name = _require_backend_key(config)
    cls = _lookup(backend_name, _content_backends(), role="content_encoder")
    return cls()


def build_speaker_encoder(config: dict[str, Any]) -> SpeakerEncoder:
    """Instantiate a :class:`~accent_converter.models.interfaces.SpeakerEncoder`."""
    backend_name = _require_backend_key(config)
    cls = _lookup(backend_name, _speaker_backends(), role="speaker_encoder")
    return cls()


def build_accent_translator(config: dict[str, Any]) -> AccentTranslator:
    """Instantiate a :class:`~accent_converter.models.interfaces.AccentTranslator`."""
    backend_name = _require_backend_key(config)
    cls = _lookup(backend_name, _accent_backends(), role="accent_translator")
    return cls()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_backend_key(config: dict[str, Any]) -> str:
    if "backend" not in config:
        raise ValueError(
            f"Configuration is missing required key 'backend'. Got: {config!r}"
        )
    return config["backend"]


def _lookup(name: str, registry: dict[str, type], role: str) -> type:
    if name not in registry:
        available = sorted(registry.keys())
        raise KeyError(
            f"Unknown {role} backend: {name!r}. "
            f"Available backends: {available}"
        )
    return registry[name]
