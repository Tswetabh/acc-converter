"""
models.backends.synthesis.vocoder
===================================
Vocoder-based waveform synthesis backend using an Acoustic Decoder and HiFi-GAN.

Architecture
------------
The synthesis engine operates in two internal stages:

::

    HuBERT / Conformer Rep (T, 768)  +  ECAPA Embedding (192,)
                        ↓
            [1] Acoustic Decoder (PyTorch)
                Maps (T, 768) + (192,) to (80, T_mel) log-mel spectrogram
                with 1.25x temporal upsampling (50 Hz -> 62.5 Hz).
                        ↓
            [2] Neural Vocoder (HiFi-GAN 16 kHz)
                ``speechbrain/tts-hifigan-libritts-16kHz`` maps mel to
                16 kHz mono float32 waveform.
                        ↓
                16 kHz mono float32 waveform
"""

from __future__ import annotations

import os
from typing import Any
import numpy as np

from accent_converter.models.interfaces import SpeechSynthesizer


class VocoderSynthesizer(SpeechSynthesizer):
    """Vocoder-based waveform synthesis backend.

    Parameters
    ----------
    checkpoint_path:
        Path to the Acoustic Decoder checkpoint weights.

    Attributes
    ----------
    _loaded : bool
        True after a successful ``load()`` call.
    _is_stub : bool
        True if running in stub mode (returns zero waveform).
    _checkpoint_path : str or None
        Path supplied to ``load()``.
    _device : str or None
        Torch device string (e.g. ``"cuda:0"`` or ``"cpu"``).
    _acoustic_decoder : Any or None
        AcousticDecoder instance if loaded.
    _vocoder : Any or None
        SpeechBrain HIFIGAN instance if loaded.
    """

    BACKEND_NAME: str = "vocoder"
    _DEFAULT_VOCODER: str = "speechbrain/tts-hifigan-libritts-16kHz"

    def __init__(self) -> None:
        self._loaded: bool = False
        self._is_stub: bool = False
        self._checkpoint_path: str | None = None
        self._device: str | None = None
        self._acoustic_decoder: Any = None
        self._vocoder: Any = None

    # ------------------------------------------------------------------
    # SpeechSynthesizer interface
    # ------------------------------------------------------------------

    def load(
        self,
        checkpoint_path: str | None = None,
        device: str | None = None,
        vocoder_source: str | None = None,
        savedir: str | None = None,
    ) -> None:
        """Load Acoustic Decoder weights and HiFi-GAN vocoder.

        Parameters
        ----------
        checkpoint_path:
            Path to the Acoustic Decoder checkpoint (.pt file).
            If None or ``"stub"``, operates in Phase 6A stub mode.
        device:
            Torch device (e.g. ``"cuda:0"`` or ``"cpu"``). If None, selects
            CUDA if available, else CPU.
        vocoder_source:
            Source identifier or directory for HiFi-GAN vocoder.
        savedir:
            Directory to cache vocoder weights.

        Raises
        ------
        FileNotFoundError:
            If a non-stub checkpoint path is provided but does not exist.
        ValueError:
            If an invalid torch device is supplied.
        RuntimeError:
            If loading fails.
        """
        self._checkpoint_path = checkpoint_path

        if checkpoint_path is None or checkpoint_path == "stub":
            self._is_stub = True
            self._loaded = True
            return

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Acoustic decoder checkpoint file not found: {checkpoint_path}"
            )

        import torch
        from speechbrain.inference.vocoders import HIFIGAN
        from speechbrain.utils.fetching import LocalStrategy
        from accent_converter.models.backends.synthesis.acoustic_decoder import AcousticDecoder

        # Device selection
        if device is None:
            selected_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            selected_device = device

        try:
            torch.device(selected_device)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"Invalid device {selected_device!r}: {exc}") from exc

        self._device = selected_device

        # 1. Load Acoustic Decoder
        self._acoustic_decoder = AcousticDecoder()
        try:
            state = torch.load(checkpoint_path, map_location=self._device)
            if isinstance(state, dict):
                if "acoustic_decoder" in state:
                    state = state["acoustic_decoder"]
                elif "model_state_dict" in state:
                    state = state["model_state_dict"]
            self._acoustic_decoder.load_state_dict(state)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load AcousticDecoder weights from {checkpoint_path}: {exc}"
            ) from exc

        self._acoustic_decoder.to(self._device).eval()

        # 2. Load HiFi-GAN vocoder
        source = vocoder_source or self._DEFAULT_VOCODER
        if savedir is None:
            # Check local models directory first
            local_models = os.path.join("models", "hifigan_16k")
            if os.path.isdir(local_models) and os.path.isfile(os.path.join(local_models, "hyperparams.yaml")):
                source = local_models
                savedir = local_models
            else:
                cache_folder = source.replace("/", "_")
                savedir = os.path.join(
                    os.path.expanduser("~"), ".cache", "speechbrain", cache_folder
                )

        run_opts = {"device": self._device}
        strategy = LocalStrategy.COPY if os.name == "nt" else LocalStrategy.SYMLINK

        try:
            self._vocoder = HIFIGAN.from_hparams(
                source=source,
                savedir=savedir,
                run_opts=run_opts,
                local_strategy=strategy,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load HiFi-GAN vocoder from {source}: {exc}"
            ) from exc

        self._is_stub = False
        self._loaded = True

    def synthesize(
        self,
        converted_rep: np.ndarray,
        speaker_embedding: np.ndarray,
    ) -> np.ndarray:
        """Synthesize a waveform from *converted_rep* and *speaker_embedding*.

        Parameters
        ----------
        converted_rep:
            Continuous content representations of shape ``(T, 768)``.
        speaker_embedding:
            Speaker identity vector of shape ``(192,)``.

        Returns
        -------
        np.ndarray
            1D float32 audio waveform sampled at 16 kHz.
        """
        if not self._loaded:
            raise RuntimeError(
                "VocoderSynthesizer.synthesize() called before load()."
            )

        if not isinstance(converted_rep, np.ndarray):
            raise TypeError(
                f"converted_rep must be a numpy ndarray, got {type(converted_rep).__name__}"
            )

        if not isinstance(speaker_embedding, np.ndarray):
            raise TypeError(
                f"speaker_embedding must be a numpy ndarray, got {type(speaker_embedding).__name__}"
            )

        if speaker_embedding.ndim != 1 or speaker_embedding.shape[0] != 192:
            raise ValueError(
                f"speaker_embedding must be 1-D with shape (192,), got {speaker_embedding.shape}"
            )

        if converted_rep.ndim != 2:
            raise ValueError(
                f"converted_rep must be 2-D with shape (T, D), got {converted_rep.shape}"
            )

        if converted_rep.shape[0] == 0:
            return np.empty(0, dtype=np.float32)

        # Stub mode: return zero waveform matching expected duration
        if self._is_stub:
            t_frames = converted_rep.shape[0]
            n_samples = t_frames * 320
            return np.zeros(n_samples, dtype=np.float32)

        # Real synthesis mode
        import torch

        rep_tensor = torch.from_numpy(converted_rep.astype(np.float32)).to(self._device)
        spk_tensor = torch.from_numpy(speaker_embedding.astype(np.float32)).to(self._device)

        with torch.inference_mode():
            # 1. Acoustic Decoder: (1, T, 768) + (1, 192) -> (1, 80, T_mel)
            mel = self._acoustic_decoder(rep_tensor, spk_tensor)

            # 2. HiFi-GAN vocoder: (1, 80, T_mel) -> (1, 1, N)
            wav_tensor = self._vocoder.decode_batch(mel)

            # 3. Convert to 1D numpy array
            wav = wav_tensor.squeeze().detach().cpu().numpy().astype(np.float32)

        return wav

    def reset(self) -> None:
        """Reset autoregressive / streaming state.

        Safe to call at any time.
        """
        pass

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"VocoderSynthesizer("
            f"loaded={self._loaded}, "
            f"stub={self._is_stub}, "
            f"checkpoint={self._checkpoint_path!r})"
        )
