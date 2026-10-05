"""
models.backends.content.hubert
================================
HuBERT-based offline content encoder backend.

Status
------
Phase 4B IMPLEMENTED.  ``load()`` and ``encode()`` are functional.

Phase
-----
Phase 4 — offline reference encoder.

Role in the pipeline
--------------------
HuBERT is the **offline reference content encoder** for Phases 4–6.

It produces linguistic/content representations intended for use in
speaker/accent disentanglement. The actual degree of speaker information
retained in the representation should be evaluated experimentally.

It is NOT a streaming encoder.  Do NOT use it in the live inference path.

Architecture
------------
* Waveform → 7-layer strided CNN feature extractor (receptive field ~25 ms)
* Feature frames → 12-layer bidirectional Transformer encoder (HuBERT Base)
* Output: dense float32 tensor, shape ``(T_frames, 768)``
* Frame rate: 50 Hz (one frame per 20 ms, from CNN stride = 320 @ 16 kHz)
* Attention: full bidirectional self-attention — non-causal, unbounded lookahead

Why it is NOT causal (documented explicitly)
--------------------------------------------
Standard HuBERT uses bidirectional Transformer attention.  The representation
of frame t depends on all frames including t+1, t+2, ..., T (future frames).

This means:
  - Calling encode(chunk) on successive 80 ms chunks produces representations
    that are *inconsistent* across calls — they would change if more audio
    arrived.
  - Correct output for frame t is only available after the entire utterance
    has been encoded.
  - HuBERT cannot produce committed outputs for a streaming session.

``lookahead_ms`` returns ``-1`` as the sentinel value for "non-causal /
unbounded lookahead".  The pipeline scheduler must check this value and
reject non-causal encoders from the live path.

Checkpoint
----------
* HuggingFace model ID: ``facebook/hubert-base-ls960``
* Parameters: ~90 M
* Checkpoint size: ~360 MB
* License: Apache 2.0

Configure via::

    models:
      content_encoder:
        backend: hubert
        checkpoint: facebook/hubert-base-ls960   # or local path

Dependencies
------------
* torch>=2.0
* transformers>=4.30
"""

from __future__ import annotations

import numpy as np

from accent_converter.models.interfaces import ContentEncoder


class HubertContentEncoder(ContentEncoder):
    """HuBERT Base offline content encoder.

    Uses ``Wav2Vec2FeatureExtractor`` and ``HubertModel`` from HuggingFace
    ``transformers`` to extract frame-level content representations from
    16 kHz mono float32 audio.

    This encoder is **non-causal** (``lookahead_ms = -1``).  It must NOT
    be placed in the live streaming inference path.

    Attributes
    ----------
    _loaded : bool
        True after a successful ``load()`` call.
    _checkpoint_path : str or None
        Checkpoint path / HuggingFace model ID supplied to ``load()``.
    _device : str or None
        Torch device string (``"cpu"``, ``"cuda"``) selected during ``load()``.
    _processor : Wav2Vec2FeatureExtractor or None
        Feature extractor instance after loading.
    _model : HubertModel or None
        Transformer model instance after loading.

    Metadata properties (fixed for HuBERT Base)
    --------------------------------------------
    frame_rate_hz : int = 50
        HuBERT CNN feature extractor produces one frame per 20 ms (50 Hz).
        CNN stride = 320 samples @ 16 kHz → 16 000 / 320 = 50 frames/s.
    output_dim : int = 768
        HuBERT Base has 12 transformer layers with hidden size 768.
    lookahead_ms : int = -1
        Sentinel: non-causal / unbounded.  This backend must NOT be used in
        the live streaming path.  Check ``lookahead_ms >= 0`` before using
        an encoder in real-time mode.
    """

    BACKEND_NAME: str = "hubert"

    # -- Metadata constants (HuBERT Base, verified empirically in Phase 4B) --
    _FRAME_RATE_HZ: int = 50       # 20 ms / frame
    _OUTPUT_DIM: int = 768         # HuBERT Base hidden size
    _LOOKAHEAD_MS: int = -1        # non-causal (unbounded); offline only
    _DEFAULT_CHECKPOINT: str = "facebook/hubert-base-ls960"

    def __init__(self) -> None:
        self._loaded: bool = False
        self._checkpoint_path: str | None = None
        self._device: str | None = None
        self._processor = None
        self._model = None

    # ------------------------------------------------------------------
    # ContentEncoder interface — model methods
    # ------------------------------------------------------------------

    def load(self, checkpoint_path: str | None = None, device: str | None = None) -> None:
        """Load HuBERT weights from *checkpoint_path*.

        Parameters
        ----------
        checkpoint_path:
            HuggingFace model ID (e.g. ``"facebook/hubert-base-ls960"``) or
            path to a local directory produced by ``model.save_pretrained()``.
            If ``None`` or empty, defaults to ``facebook/hubert-base-ls960``.
        device:
            Torch device to run inference on.  If ``None`` (default), CUDA is
            selected automatically when ``torch.cuda.is_available()``, with
            CPU as fallback.  Pass ``"cpu"`` or ``"cuda"`` to override.

        Raises
        ------
        ValueError
            If *device* is supplied but is not a recognised torch device.
        RuntimeError
            If the model or processor cannot be loaded (e.g. network error,
            missing checkpoint).
        """
        # Deferred imports: torch and transformers are only needed at load time.
        import torch
        from transformers import Wav2Vec2FeatureExtractor, HubertModel

        # -- Device selection --
        if device is None:
            selected_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            selected_device = device

        # Validate the device string by constructing a torch.device object.
        try:
            torch.device(selected_device)
        except RuntimeError as exc:
            raise ValueError(
                f"Invalid device {selected_device!r}: {exc}"
            ) from exc

        if checkpoint_path is None or checkpoint_path == "":
            source = self._DEFAULT_CHECKPOINT
        else:
            source = checkpoint_path

        # -- Load processor (Wav2Vec2FeatureExtractor) --
        processor = Wav2Vec2FeatureExtractor.from_pretrained(source)

        # -- Load model (HubertModel) --
        model = HubertModel.from_pretrained(source)
        model.eval()
        model = model.to(selected_device)

        # -- Store state (only after successful loading) --
        self._processor = processor
        self._model = model
        self._checkpoint_path = source
        self._device = selected_device
        self._loaded = True

    def encode(self, audio: np.ndarray) -> np.ndarray:
        """Encode *audio* to a HuBERT content representation.

        Parameters
        ----------
        audio:
            1-D NumPy array of float32 audio samples at 16 000 Hz.
            Must be non-empty and contain only finite values.

        Returns
        -------
        np.ndarray
            Float32 array of shape ``(T_frames, 768)`` at 50 Hz frame rate.
            ``T_frames`` depends on the audio duration; do not assume a fixed
            count per second.

        Raises
        ------
        RuntimeError
            If called before ``load()``.
        ValueError
            If *audio* is not a 1-D float32 ndarray, is empty, or contains
            non-finite values (inf / nan).
        AssertionError
            Internal guard: if the model unexpectedly returns a feature
            dimension other than 768.

        Notes
        -----
        This encoder is non-causal.  The output depends on the full input
        sequence.  Do not use in the streaming inference path.
        """
        import torch

        # -- Loaded-state guard --
        if not self._loaded:
            raise RuntimeError(
                "HubertContentEncoder.encode() called before load(). "
                "Call load(checkpoint_path) first."
            )

        # -- Input validation --
        if not isinstance(audio, np.ndarray):
            raise ValueError(
                f"audio must be a numpy.ndarray, got {type(audio).__name__!r}."
            )
        if audio.ndim != 1:
            raise ValueError(
                f"audio must be 1-D (mono), got shape {audio.shape}."
            )
        if len(audio) == 0:
            raise ValueError("audio must not be empty.")
        if audio.dtype != np.float32:
            raise ValueError(
                f"audio dtype must be float32, got {audio.dtype}. "
                "Convert with audio.astype(np.float32) before calling encode()."
            )
        if not np.all(np.isfinite(audio)):
            raise ValueError(
                "audio contains non-finite values (inf or nan). "
                "Apply audio normalisation before calling encode()."
            )

        # -- Feature extraction --
        # The processor returns a dict of PyTorch-ready tensors.
        inputs = self._processor(
            audio,
            sampling_rate=16_000,
            return_tensors="pt",
        )
        input_values = inputs["input_values"].to(self._device)

        # -- Inference --
        with torch.inference_mode():
            outputs = self._model(input_values)

        # outputs.last_hidden_state: (1, T_frames, 768)
        features = outputs.last_hidden_state.squeeze(0)  # → (T_frames, 768)

        # -- Move to CPU and convert to NumPy --
        features_np = features.cpu().to(torch.float32).numpy()

        # -- Output dimension guard --
        assert features_np.shape[-1] == self._OUTPUT_DIM, (
            f"Expected output_dim={self._OUTPUT_DIM}, "
            f"but HuBERT returned last dim={features_np.shape[-1]}. "
            "This indicates a checkpoint/model mismatch."
        )

        return features_np

    def reset(self) -> None:
        """No-op for HuBERT — the model is stateless for offline use.

        HuBERT Base has no KV cache, ring buffer, or streaming state.
        Safe to call at any time (before or after load()).
        """
        pass

    # ------------------------------------------------------------------
    # ContentEncoder interface — metadata properties
    # ------------------------------------------------------------------

    @property
    def frame_rate_hz(self) -> int:
        """50 Hz — one output frame per 20 ms.

        Determined by the HuBERT CNN feature extractor stride:
        7 convolutions with total stride of 320 samples at 16 kHz
        → 16 000 / 320 = 50 frames/second.
        """
        return self._FRAME_RATE_HZ

    @property
    def output_dim(self) -> int:
        """768 — HuBERT Base Transformer hidden size."""
        return self._OUTPUT_DIM

    @property
    def lookahead_ms(self) -> int:
        """-1 — non-causal sentinel.

        HuBERT uses full bidirectional self-attention with unbounded lookahead.
        This model must NOT be placed in the live streaming inference path.
        Check ``lookahead_ms >= 0`` before using an encoder in streaming mode.
        """
        return self._LOOKAHEAD_MS

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"HubertContentEncoder("
            f"loaded={self._loaded}, "
            f"checkpoint={self._checkpoint_path!r}, "
            f"device={self._device!r}, "
            f"frame_rate_hz={self.frame_rate_hz}, "
            f"output_dim={self.output_dim}, "
            f"lookahead_ms={self.lookahead_ms})"
        )
