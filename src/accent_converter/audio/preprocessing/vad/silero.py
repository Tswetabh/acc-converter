"""
audio.preprocessing.vad.silero
==============================
Neural Voice Activity Detector (VAD) backend powered by Silero VAD.

Silero VAD is an enterprise-grade, lightweight pre-trained neural network
designed for fast, robust speech/non-speech classification. Unlike pure energy
detectors, it learns acoustic speech features, making it highly resilient to
loud non-speech noises (keyboard clicks, fans, music, static, transient pops).

Algorithm & Streaming Integration
---------------------------------
1. Silero VAD evaluates speech probabilities on 512-sample audio windows at
   16 kHz (32 ms) or 256-sample windows at 8 kHz.
2. The pipeline passes audio chunks of arbitrary length (canonically 80 ms =
   1,280 samples at 16 kHz).
3. ``SileroVAD`` maintains an internal streaming buffer:
   - Incoming audio samples are appended to the buffer.
   - For every complete 512-sample window available in the buffer, the neural
     network computes a speech probability in ``[0.0, 1.0]``.
   - If any window evaluated during the chunk yields a probability >=
     ``threshold``, the chunk is tentatively classified as speech.
4. A configurable **hangover** counter is applied:
   - Once speech is detected, ``is_speech=True`` is held for at least
     ``hangover_frames`` additional chunks.
   - This prevents rapid toggling and syllable clipping during natural
     intra-sentence pauses.
5. Timing is strictly preserved: the original audio array is returned
   **unmodified** in ``VADResult.audio``.
6. Energy in dB is computed via RMS for diagnostic continuity.

Backend name: ``"silero"``

Configuration
-------------
::

    preprocessing:
      vad:
        backend: silero
        threshold: 0.5         # probability threshold in [0.0, 1.0] (default: 0.5)
        hangover_frames: 8     # frames to hold speech after detection (8 × 80 ms = 640 ms)
        device: null           # null -> auto ('cuda' if available else 'cpu')
"""

from __future__ import annotations

from typing import Any
import numpy as np
import torch

from accent_converter.audio.preprocessing.interfaces import (
    VoiceActivityDetector,
    VADResult,
)
from accent_converter.audio.preprocessing.vad.energy import _rms_db, _FLOOR_DB

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SILERO_THRESHOLD: float = 0.5
DEFAULT_HANGOVER_FRAMES: int = 8
WINDOW_SIZE_16K: int = 512
WINDOW_SIZE_8K: int = 256


# ---------------------------------------------------------------------------
# SileroVAD
# ---------------------------------------------------------------------------


class SileroVAD(VoiceActivityDetector):
    """Neural Voice Activity Detector backend using Silero VAD.

    Parameters
    ----------
    threshold:
        Probability threshold in [0.0, 1.0]. Audio windows with predicted
        speech probability >= threshold are classified as speech.
        Default: ``0.5``.
    hangover_frames:
        Number of consecutive non-speech frames to hold a ``is_speech=True``
        label after the last speech-positive frame. Default: 8
        (640 ms at 80 ms chunks).
    sampling_rate:
        Audio sample rate. Supported values: 16000 or 8000 Hz. Default: 16000.
    device:
        Torch device ('cpu', 'cuda', etc.). If None, automatically selects
        'cuda' if available, otherwise 'cpu'.
    model:
        Optional pre-instantiated or mocked Silero model (used for unit testing
        to bypass neural network initialization).

    Attributes
    ----------
    threshold : float
    hangover_frames : int
    sampling_rate : int
    device : torch.device
    """

    BACKEND_NAME: str = "silero"

    def __init__(
        self,
        threshold: float = DEFAULT_SILERO_THRESHOLD,
        hangover_frames: int = DEFAULT_HANGOVER_FRAMES,
        sampling_rate: int = 16000,
        device: str | None = None,
        model: Any | None = None,
    ) -> None:
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(
                f"threshold must be between 0.0 and 1.0, got {threshold}"
            )
        if hangover_frames < 0:
            raise ValueError(
                f"hangover_frames must be >= 0, got {hangover_frames}"
            )
        if sampling_rate not in (8000, 16000):
            raise ValueError(
                f"sampling_rate must be 8000 or 16000 Hz, got {sampling_rate}"
            )

        self.threshold: float = float(threshold)
        self.hangover_frames: int = int(hangover_frames)
        self.sampling_rate: int = int(sampling_rate)
        self.window_size: int = (
            WINDOW_SIZE_16K if self.sampling_rate == 16000 else WINDOW_SIZE_8K
        )

        if device is None:
            self.device: torch.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        if model is None:
            import silero_vad
            self._model = silero_vad.load_silero_vad()
            self._model.eval()
            self._model.to(self.device)
        else:
            self._model = model

        self._buffer: np.ndarray = np.empty(0, dtype=np.float32)
        self._hangover_counter: int = 0
        self._last_speech_prob: float = 0.0

    # ------------------------------------------------------------------
    # VoiceActivityDetector interface
    # ------------------------------------------------------------------

    def process(self, chunk: np.ndarray) -> VADResult:
        """Classify *chunk* as speech or non-speech using Silero VAD.

        Parameters
        ----------
        chunk:
            1-D float32 array, canonical 16 kHz format.

        Returns
        -------
        VADResult
            ``audio`` is the original chunk, **unmodified**.
            ``is_speech`` is True if speech was detected (including hangover).
            ``energy_db`` is the measured short-time RMS energy.
        """
        if not isinstance(chunk, np.ndarray):
            raise TypeError(
                f"chunk must be a numpy ndarray, got {type(chunk).__name__}"
            )

        if chunk.ndim != 1:
            raise ValueError(
                f"chunk must be 1-D, got shape {chunk.shape}"
            )

        if len(chunk) == 0:
            return VADResult(audio=chunk, is_speech=False, energy_db=_FLOOR_DB)

        if not np.issubdtype(chunk.dtype, np.floating):
            raise TypeError(
                f"chunk must have floating-point dtype, got {chunk.dtype}"
            )

        if not np.all(np.isfinite(chunk)):
            raise ValueError("chunk contains non-finite values (NaN or Inf)")

        chunk_f32 = (
            chunk if chunk.dtype == np.float32 else chunk.astype(np.float32)
        )
        energy_db = _rms_db(chunk_f32)

        # Buffer samples for windowing
        self._buffer = np.concatenate([self._buffer, chunk_f32])

        window_probs: list[float] = []
        while len(self._buffer) >= self.window_size:
            window = self._buffer[:self.window_size]
            self._buffer = self._buffer[self.window_size:]

            tensor_win = torch.from_numpy(window).to(self.device)
            with torch.inference_mode():
                prob = float(self._model(tensor_win, self.sampling_rate).item())
            window_probs.append(prob)

        if window_probs:
            max_prob = max(window_probs)
            self._last_speech_prob = max_prob
            raw_speech = max_prob >= self.threshold
        else:
            raw_speech = self._last_speech_prob >= self.threshold

        if raw_speech:
            self._hangover_counter = self.hangover_frames
            is_speech = True
        elif self._hangover_counter > 0:
            self._hangover_counter -= 1
            is_speech = True
        else:
            is_speech = False

        return VADResult(audio=chunk, is_speech=is_speech, energy_db=energy_db)

    def reset(self) -> None:
        """Reset internal streaming state, buffer, and hangover counter."""
        if hasattr(self._model, "reset_states"):
            self._model.reset_states()
        self._buffer = np.empty(0, dtype=np.float32)
        self._hangover_counter = 0
        self._last_speech_prob = 0.0

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SileroVAD("
            f"threshold={self.threshold}, "
            f"hangover_frames={self.hangover_frames}, "
            f"device={self.device.type}, "
            f"_hangover_counter={self._hangover_counter})"
        )
