"""
audio.preprocessing.aec.passthrough
=====================================
Passthrough AEC backend — for use until browser/WebRTC AEC is integrated.

Why a passthrough?
------------------
Acoustic Echo Cancellation requires a **far-end reference** signal (what the
loudspeaker is playing) to adaptively subtract the echo from the microphone
signal.  In a browser-based voice call this reference is available to the
WebRTC stack natively.

The project's canonical AEC implementation is **browser-side WebRTC AEC3**,
which runs in the browser before audio is sent to the backend.  There is no
server-side Python AEC in Phase 3.

This ``PassthroughAEC`` class:
  - Satisfies the ``AcousticEchoCanceller`` interface so the pipeline can be
    fully wired today.
  - Returns the near-end signal unchanged (no echo removal performed).
  - Logs a warning when ``far_end`` is supplied, to catch accidental misuse.
  - Documents exactly what must happen in Phase 11 to replace it.

Phase 11 integration plan
--------------------------
When browser WebRTC integration is complete:
  1. Verify that the browser's getUserMedia() MediaStreamTrack has
     ``echoCancellation: true`` in the AudioContext constraints.
  2. Confirm that the audio arriving at the Python backend is already
     echo-cancelled by the browser.
  3. If server-side AEC is still needed (e.g. for non-browser clients):
     - Evaluate ``speexdsp`` Python bindings or a custom Cython wrapper
       around WebRTC AEC.
     - Implement a new ``WebRTCAEC`` or ``SpeexAEC`` backend.
     - Register it in ``preprocessing/factory.py``.
     - The pipeline requires zero changes.

Backend name: ``"passthrough"``
"""

from __future__ import annotations

import warnings

import numpy as np

from accent_converter.audio.preprocessing.interfaces import AcousticEchoCanceller


class PassthroughAEC(AcousticEchoCanceller):
    """Passthrough AEC — returns near-end unchanged; no echo removal.

    The pipeline slot for AEC is correctly wired.  Real AEC will be
    provided by browser-side WebRTC (Phase 11) or a future server-side
    backend.

    Attributes
    ----------
    _warn_on_far_end : bool
        If True (default), emit a warning when ``far_end`` is supplied,
        because it is being silently ignored.
    """

    BACKEND_NAME: str = "passthrough"

    def __init__(self, warn_on_far_end: bool = True) -> None:
        self._warn_on_far_end = warn_on_far_end

    # ------------------------------------------------------------------
    # AcousticEchoCanceller interface
    # ------------------------------------------------------------------

    def process(
        self,
        near_end: np.ndarray,
        far_end: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return *near_end* unchanged — no echo cancellation performed.

        Parameters
        ----------
        near_end:
            1-D float32 microphone signal.
        far_end:
            Ignored.  A warning is emitted if provided, because real AEC
            would use this and its absence indicates a wiring issue.

        Returns
        -------
        np.ndarray
            Copy of *near_end* (unmodified).
        """
        if near_end.ndim != 1:
            raise ValueError(
                f"near_end must be 1-D, got shape {near_end.shape}"
            )
        if far_end is not None and self._warn_on_far_end:
            warnings.warn(
                "PassthroughAEC received a far_end reference but is not "
                "performing echo cancellation.  The far_end signal is being "
                "ignored.  Replace PassthroughAEC with a real AEC backend "
                "or rely on browser-side WebRTC AEC (Phase 11).",
                stacklevel=2,
            )
        return near_end.copy()

    def reset(self) -> None:
        """No-op — no adaptive filter state to reset."""
        pass

    def __repr__(self) -> str:  # pragma: no cover
        return "PassthroughAEC()"
