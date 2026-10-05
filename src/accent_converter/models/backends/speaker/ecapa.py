"""
models.backends.speaker.ecapa
=============================
SpeechBrain ECAPA-TDNN speaker encoder backend.

Extracts a fixed 192-dimensional speaker embedding from reference/enrollment
audio using ``speechbrain/spkrec-ecapa-voxceleb``.

Key properties
--------------
- Input: 16 kHz mono float32 audio.
- Output: (192,) float32 NumPy array.
- Stateless: Speaker embeddings are computed once from enrollment audio
  and cached across streaming chunks.
"""

from __future__ import annotations

import os
from typing import Any
import numpy as np

from accent_converter.models.interfaces import SpeakerEncoder


class EcapaSpeakerEncoder(SpeakerEncoder):
    """ECAPA-TDNN speaker encoder backend using SpeechBrain.

    Attributes
    ----------
    _loaded : bool
        True after a successful ``load()`` call.
    _checkpoint_path : str or None
        Model ID or local directory used for loading.
    _device : str or None
        Torch device used for inference.
    _classifier : Any or None
        SpeechBrain EncoderClassifier instance.
    """

    BACKEND_NAME: str = "ecapa"
    _EMBEDDING_DIM: int = 192
    _DEFAULT_CHECKPOINT: str = "speechbrain/spkrec-ecapa-voxceleb"

    def __init__(self) -> None:
        self._loaded: bool = False
        self._checkpoint_path: str | None = None
        self._device: str | None = None
        self._classifier: Any = None

    def load(
        self,
        checkpoint_path: str | None = None,
        device: str | None = None,
        savedir: str | None = None,
    ) -> None:
        """Load ECAPA-TDNN model weights from *checkpoint_path*.

        Parameters
        ----------
        checkpoint_path:
            HuggingFace model ID (e.g. ``"speechbrain/spkrec-ecapa-voxceleb"``)
            or path to a local directory. If None or empty, defaults to
            ``"speechbrain/spkrec-ecapa-voxceleb"``.
        device:
            Torch device to run inference on (e.g. ``"cpu"``, ``"cuda"``).
            If None, automatically selects CUDA if available, else CPU.
        savedir:
            Local directory to cache downloaded weights. If None, caches in
            ``~/.cache/speechbrain/spkrec-ecapa-voxceleb``.

        Raises
        ------
        ValueError
            If *device* is supplied but is not a recognized torch device.
        RuntimeError
            If model weights cannot be loaded.
        """
        import torch
        from speechbrain.inference.classifiers import EncoderClassifier

        if checkpoint_path is None or checkpoint_path == "":
            source = self._DEFAULT_CHECKPOINT
        else:
            source = checkpoint_path

        # Device selection
        if device is None:
            selected_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            selected_device = device

        try:
            torch.device(selected_device)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(
                f"Invalid device {selected_device!r}: {exc}"
            ) from exc

        if savedir is None:
            cache_folder = source.replace("/", "_")
            savedir = os.path.join(
                os.path.expanduser("~"), ".cache", "speechbrain", cache_folder
            )

        run_opts = {"device": selected_device}
        try:
            # On Windows without Developer Mode, creating symlinks can fail with WinError 1314.
            # Use LocalStrategy.COPY if on Windows (os.name == 'nt') or as fallback.
            from speechbrain.utils.fetching import LocalStrategy
            strategy = LocalStrategy.COPY if os.name == "nt" else LocalStrategy.SYMLINK
            classifier = EncoderClassifier.from_hparams(
                source=source,
                savedir=savedir,
                run_opts=run_opts,
                local_strategy=strategy,
            )
        except OSError:
            from speechbrain.utils.fetching import LocalStrategy
            classifier = EncoderClassifier.from_hparams(
                source=source,
                savedir=savedir,
                run_opts=run_opts,
                local_strategy=LocalStrategy.COPY,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load ECAPA-TDNN model from {source!r}: {exc}"
            ) from exc

        self._classifier = classifier
        self._checkpoint_path = source
        self._device = selected_device
        self._loaded = True

    def encode(self, reference_audio: np.ndarray) -> np.ndarray:
        """Compute a 192-dimensional speaker embedding from *reference_audio*.

        Parameters
        ----------
        reference_audio:
            1-D float32 array, canonical 16 kHz format. Must be non-empty
            and contain only finite values.

        Returns
        -------
        np.ndarray
            1-D float32 NumPy array of shape ``(192,)``.

        Raises
        ------
        RuntimeError
            If called before ``load()``.
        TypeError
            If *reference_audio* is not a numpy array or is not float32.
        ValueError
            If *reference_audio* is not 1-D, is empty, or contains inf/nan.
        AssertionError
            If the model output dimension is not 192.
        """
        import torch

        if not self._loaded:
            raise RuntimeError(
                "EcapaSpeakerEncoder.encode() called before load(). "
                "Call load() first."
            )

        if not isinstance(reference_audio, np.ndarray):
            raise TypeError(
                f"reference_audio must be a numpy.ndarray, got {type(reference_audio).__name__!r}."
            )
        if reference_audio.ndim != 1:
            raise ValueError(
                f"reference_audio must be 1-D (mono), got shape {reference_audio.shape}."
            )
        if len(reference_audio) == 0:
            raise ValueError("reference_audio must not be empty.")
        if reference_audio.dtype != np.float32:
            raise TypeError(
                f"reference_audio dtype must be float32, got {reference_audio.dtype}. "
                "Convert with reference_audio.astype(np.float32) before calling encode()."
            )
        if not np.all(np.isfinite(reference_audio)):
            raise ValueError(
                "reference_audio contains non-finite values (inf or nan)."
            )

        # Batch of 1: [1, time]
        wavs = torch.from_numpy(reference_audio).unsqueeze(0).to(self._device)

        with torch.inference_mode():
            outputs = self._classifier.encode_batch(wavs)

        # outputs can be [1, 1, 192] or [1, 192]
        emb_np = outputs.squeeze().detach().cpu().numpy().astype(np.float32)

        if emb_np.ndim != 1 or emb_np.shape[0] != self._EMBEDDING_DIM:
            raise AssertionError(
                f"Expected speaker embedding shape ({self._EMBEDDING_DIM},), "
                f"got {emb_np.shape}."
            )

        return emb_np

    def reset(self) -> None:
        """Reset internal state.

        Speaker encoders are stateless across chunks; this is a safe no-op.
        """
        pass

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the speaker embedding vector (192)."""
        return self._EMBEDDING_DIM
