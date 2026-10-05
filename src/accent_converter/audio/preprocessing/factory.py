"""
audio.preprocessing.factory
============================
Factory functions for the three audio preprocessing components.

Usage
-----
::

    from accent_converter.audio.preprocessing.factory import (
        build_vad,
        build_noise_suppressor,
        build_aec,
    )

    vad = build_vad(config["preprocessing"]["vad"])
    ns  = build_noise_suppressor(config["preprocessing"]["noise_suppressor"])
    aec = build_aec(config["preprocessing"]["aec"])

Configuration keys
------------------
Each function expects a sub-dict with at least a ``"backend"`` key.

Adding a new backend
--------------------
1. Create ``audio/preprocessing/<role>/<name>.py``.
2. Subclass the appropriate ABC from ``interfaces.py``.
3. Add a ``BACKEND_NAME`` class attribute.
4. Register the class in the relevant ``_*_backends()`` dict below.
5. The pipeline and tests require no changes.
"""

from __future__ import annotations

from typing import Any

from accent_converter.audio.preprocessing.interfaces import (
    VoiceActivityDetector,
    NoiseSuppressor,
    AcousticEchoCanceller,
)


# ---------------------------------------------------------------------------
# Backend registries (deferred imports)
# ---------------------------------------------------------------------------


def _vad_backends() -> dict[str, type[VoiceActivityDetector]]:
    from accent_converter.audio.preprocessing.vad.energy import EnergyVAD
    from accent_converter.audio.preprocessing.vad.silero import SileroVAD
    return {
        EnergyVAD.BACKEND_NAME: EnergyVAD,
        SileroVAD.BACKEND_NAME: SileroVAD,
    }


def _noise_backends() -> dict[str, type[NoiseSuppressor]]:
    from accent_converter.audio.preprocessing.noise.noisereduce_backend import (
        NoiseReduceNoiseSuppressor,
    )
    return {
        NoiseReduceNoiseSuppressor.BACKEND_NAME: NoiseReduceNoiseSuppressor,
    }


def _aec_backends() -> dict[str, type[AcousticEchoCanceller]]:
    from accent_converter.audio.preprocessing.aec.passthrough import PassthroughAEC
    return {
        PassthroughAEC.BACKEND_NAME: PassthroughAEC,
    }


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def build_vad(config: dict[str, Any]) -> VoiceActivityDetector:
    """Instantiate a :class:`~...interfaces.VoiceActivityDetector`.

    Parameters
    ----------
    config:
        The ``preprocessing.vad`` sub-dict from the project config::

            {"backend": "energy", "threshold_db": -40, "hangover_frames": 8}

    Returns
    -------
    VoiceActivityDetector

    Raises
    ------
    ValueError
        If ``"backend"`` key is missing.
    KeyError
        If the backend name is not registered.
    """
    name = _require_backend(config)
    cls = _lookup(name, _vad_backends(), role="vad")
    kwargs = {k: v for k, v in config.items() if k != "backend"}
    return cls(**kwargs)


def build_noise_suppressor(config: dict[str, Any]) -> NoiseSuppressor:
    """Instantiate a :class:`~...interfaces.NoiseSuppressor`."""
    name = _require_backend(config)
    cls = _lookup(name, _noise_backends(), role="noise_suppressor")
    kwargs = {k: v for k, v in config.items() if k != "backend"}
    return cls(**kwargs)


def build_aec(config: dict[str, Any]) -> AcousticEchoCanceller:
    """Instantiate a :class:`~...interfaces.AcousticEchoCanceller`."""
    name = _require_backend(config)
    cls = _lookup(name, _aec_backends(), role="aec")
    kwargs = {k: v for k, v in config.items() if k != "backend"}
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_backend(config: dict[str, Any]) -> str:
    if "backend" not in config:
        raise ValueError(
            f"Preprocessing config is missing required key 'backend'. Got: {config!r}"
        )
    return config["backend"]


def _lookup(name: str, registry: dict[str, type], role: str) -> type:
    if name not in registry:
        raise KeyError(
            f"Unknown {role} backend: {name!r}. "
            f"Available: {sorted(registry.keys())}"
        )
    return registry[name]
