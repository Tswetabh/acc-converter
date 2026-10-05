"""
models.backends.synthesis.tvtsyn
==================================
Stub backend: TVTSyn-style causal waveform decoder.

Status
------
NOT IMPLEMENTED.  Raises ``NotImplementedError`` for all model methods.

Background
----------
TVTSyn refers to a streaming accent-conversion architecture that uses a
causal content encoder and a streaming waveform decoder, allowing bounded
look-back context with no future lookahead.

This backend is the **experimental alternative** to the vocoder backend.
It is provided so that the pipeline can swap synthesis backends via
configuration without modifying pipeline code.

Expected concrete implementation
---------------------------------
To be determined after researching:
  - whether an open TVTSyn implementation is available,
  - its checkpoint format,
  - its input/output tensor shapes,
  - its licensing.

Do not implement until Phase 8 research is complete.

Dependencies
------------
No ML frameworks are imported here until the implementation phase.
"""

from __future__ import annotations

import numpy as np

from accent_converter.models.interfaces import SpeechSynthesizer


class TVTSynSynthesizer(SpeechSynthesizer):
    """TVTSyn-style causal waveform decoder backend.

    Experimental alternative to ``VocoderSynthesizer``.  The pipeline selects
    this backend via configuration::

        synthesis:
          backend: tvtsyn
          checkpoint: path/to/tvtsyn.ckpt

    Attributes
    ----------
    _loaded : bool
    _checkpoint_path : str or None
    """

    BACKEND_NAME: str = "tvtsyn"

    def __init__(self) -> None:
        self._loaded: bool = False
        self._checkpoint_path: str | None = None

    # ------------------------------------------------------------------
    # SpeechSynthesizer interface
    # ------------------------------------------------------------------

    def load(self, checkpoint_path: str) -> None:
        """Load TVTSyn decoder weights from *checkpoint_path*.

        Not implemented.

        Raises
        ------
        NotImplementedError
        """
        self._checkpoint_path = checkpoint_path
        # TODO (Phase 8): load TVTSyn decoder weights after research.
        raise NotImplementedError(
            "TVTSynSynthesizer.load() is not implemented. "
            "Research TVTSyn availability and checkpoint format before implementing."
        )

    def synthesize(
        self,
        converted_rep: np.ndarray,
        speaker_embedding: np.ndarray,
    ) -> np.ndarray:
        """Synthesize a waveform from *converted_rep* and *speaker_embedding*.

        Not implemented.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            "TVTSynSynthesizer.synthesize() is not implemented."
        )

    def reset(self) -> None:
        """Reset causal streaming state.

        No-op until the model is implemented.  Safe to call at any time.
        """
        # TODO (Phase 8): reset causal decoder state between utterances.
        pass

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"TVTSynSynthesizer("
            f"loaded={self._loaded}, "
            f"checkpoint={self._checkpoint_path!r})"
        )
