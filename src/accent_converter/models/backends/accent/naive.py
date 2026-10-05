"""
models.backends.accent.naive
================================
Stub backend: passthrough/naive accent translator.

Status
------
NOT IMPLEMENTED.

Notes
-----
A real accent translator transforms the content representation to shift
accent/pronunciation while preserving speaker identity and linguistic content.
It must not be confused with an ASR system or a generic TTS model.

The architecture (seq2seq, flow-based, diffusion, etc.) is to be determined
by research in Phase 6.

Configure via::

    models:
      accent_translator:
        backend: naive
        checkpoint: path/to/translator.ckpt
        target_accent: us
"""

from __future__ import annotations

import numpy as np

from accent_converter.models.interfaces import AccentTranslator


class NaiveAccentTranslator(AccentTranslator):
    """Passthrough / naive accent translator backend stub.

    In its stub form this is a passthrough (returns the input content
    representation unchanged).  It exists to allow the pipeline to be
    wired end-to-end before the real translator is implemented.

    Attributes
    ----------
    _loaded : bool
    _checkpoint_path : str or None
    """

    BACKEND_NAME: str = "naive"

    def __init__(self) -> None:
        self._loaded: bool = False
        self._checkpoint_path: str | None = None

    def load(self, checkpoint_path: str) -> None:
        """Load translator weights. Not implemented for naive backend."""
        self._checkpoint_path = checkpoint_path
        # Naive backend has no weights — this is intentionally a no-op
        # so end-to-end pipeline tests can run without a real checkpoint.
        self._loaded = True

    def translate(
        self,
        content_rep: np.ndarray,
        target_accent: str,
    ) -> np.ndarray:
        """Passthrough: return content_rep unchanged (no real translation).

        Speaker embedding conditioning is the Synthesizer's responsibility,
        not the Translator's.  This stub ignores target_accent and returns
        the content representation unchanged to allow pipeline wiring tests.
        Replace with a real translator in Phase 6B.
        """
        return content_rep.copy()

    def reset(self) -> None:
        """No-op for the naive backend."""
        pass
