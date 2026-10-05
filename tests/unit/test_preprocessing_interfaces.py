"""
tests/unit/test_preprocessing_interfaces.py
=============================================
Interface and backend tests for accent_converter.audio.preprocessing

Covers:
- ABC enforcement (cannot instantiate without full implementation)
- Pipeline decoupling (pipeline accepts any VoiceActivityDetector / etc.)
- Factory: build_vad, build_noise_suppressor, build_aec
- Factory error handling (unknown backend, missing backend key)
- EnergyVAD:
    - speech detection above threshold
    - silence detection below threshold
    - hangover hold-on (speech held for hangover_frames after last speech)
    - hangover expiry (silence after hangover exhausted)
    - reset() clears hangover counter
    - audio is never modified (timing preserved)
    - energy_db accuracy
    - empty chunk handling
    - custom threshold and hangover
- NoiseReduceNoiseSuppressor:
    - output shape matches input
    - output dtype is float32
    - reset() is safe
    - too-short signal passthrough
- PassthroughAEC:
    - returns copy of near_end unchanged
    - warns when far_end supplied
    - no warning when warn_on_far_end=False
    - 2-D input raises ValueError
    - reset() is safe
- All three reset() safe before any other call
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from accent_converter.audio.preprocessing.interfaces import (
    VoiceActivityDetector,
    NoiseSuppressor,
    AcousticEchoCanceller,
    VADResult,
)
from accent_converter.audio.preprocessing.factory import (
    build_vad,
    build_noise_suppressor,
    build_aec,
)
from accent_converter.audio.preprocessing.vad.energy import (
    EnergyVAD,
    _rms_db,
    DEFAULT_THRESHOLD_DB,
    DEFAULT_HANGOVER_FRAMES,
)
from accent_converter.audio.preprocessing.vad.silero import SileroVAD
from accent_converter.audio.preprocessing.noise.noisereduce_backend import (
    NoiseReduceNoiseSuppressor,
)
from accent_converter.audio.preprocessing.aec.passthrough import PassthroughAEC
from accent_converter.audio.normalizer import CANONICAL_SAMPLE_RATE
from accent_converter.audio.chunker import CHUNK_SAMPLES


# ===========================================================================
# Helpers
# ===========================================================================


def _sine(freq: float = 440.0, n: int = CHUNK_SAMPLES, amp: float = 0.5) -> np.ndarray:
    """Return a float32 sine wave at 16 kHz."""
    t = np.arange(n, dtype=np.float32) / CANONICAL_SAMPLE_RATE
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def _silence(n: int = CHUNK_SAMPLES) -> np.ndarray:
    """Return a zero-amplitude float32 array."""
    return np.zeros(n, dtype=np.float32)


def _very_quiet(n: int = CHUNK_SAMPLES, amp: float = 1e-6) -> np.ndarray:
    """Return a nearly-silent float32 array."""
    return np.full(n, amp, dtype=np.float32)


# ===========================================================================
# ABC enforcement
# ===========================================================================


class TestABCEnforcement:
    def test_vad_cannot_be_instantiated_directly(self):
        with pytest.raises(TypeError):
            VoiceActivityDetector()  # type: ignore[abstract]

    def test_noise_suppressor_cannot_be_instantiated_directly(self):
        with pytest.raises(TypeError):
            NoiseSuppressor()  # type: ignore[abstract]

    def test_aec_cannot_be_instantiated_directly(self):
        with pytest.raises(TypeError):
            AcousticEchoCanceller()  # type: ignore[abstract]

    def test_partial_vad_cannot_be_instantiated(self):
        class _Partial(VoiceActivityDetector):
            def process(self, chunk): ...
            # reset() missing

        with pytest.raises(TypeError):
            _Partial()  # type: ignore[abstract]


# ===========================================================================
# Factory
# ===========================================================================


class TestFactory:
    def test_build_vad_energy(self):
        vad = build_vad({"backend": "energy"})
        assert isinstance(vad, VoiceActivityDetector)
        assert isinstance(vad, EnergyVAD)

    def test_build_vad_energy_with_kwargs(self):
        vad = build_vad({"backend": "energy", "threshold_db": -35.0, "hangover_frames": 4})
        assert isinstance(vad, EnergyVAD)
        assert vad.threshold_db == -35.0
        assert vad.hangover_frames == 4

    def test_build_vad_silero(self):
        vad = build_vad({"backend": "silero"})
        assert isinstance(vad, VoiceActivityDetector)
        assert isinstance(vad, SileroVAD)

    def test_build_vad_silero_with_kwargs(self):
        vad = build_vad({"backend": "silero", "threshold": 0.6, "hangover_frames": 2})
        assert isinstance(vad, SileroVAD)
        assert vad.threshold == 0.6
        assert vad.hangover_frames == 2

    def test_build_noise_suppressor(self):
        ns = build_noise_suppressor({"backend": "noisereduce"})
        assert isinstance(ns, NoiseSuppressor)
        assert isinstance(ns, NoiseReduceNoiseSuppressor)

    def test_build_aec_passthrough(self):
        aec = build_aec({"backend": "passthrough"})
        assert isinstance(aec, AcousticEchoCanceller)
        assert isinstance(aec, PassthroughAEC)

    def test_unknown_vad_backend_raises(self):
        with pytest.raises(KeyError, match="Unknown vad backend"):
            build_vad({"backend": "webrtc"})

    def test_unknown_noise_backend_raises(self):
        with pytest.raises(KeyError, match="Unknown noise_suppressor backend"):
            build_noise_suppressor({"backend": "rnnoise"})

    def test_unknown_aec_backend_raises(self):
        with pytest.raises(KeyError, match="Unknown aec backend"):
            build_aec({"backend": "speex"})

    def test_missing_backend_key_raises_value_error(self):
        with pytest.raises(ValueError, match="missing required key 'backend'"):
            build_vad({"threshold_db": -40})

    def test_factory_returns_correct_abc_type(self):
        """All factory functions return the expected interface type."""
        assert isinstance(build_vad({"backend": "energy"}), VoiceActivityDetector)
        assert isinstance(build_noise_suppressor({"backend": "noisereduce"}), NoiseSuppressor)
        assert isinstance(build_aec({"backend": "passthrough"}), AcousticEchoCanceller)


# ===========================================================================
# EnergyVAD
# ===========================================================================


class TestEnergyVADConstruction:
    def test_default_parameters(self):
        vad = EnergyVAD()
        assert vad.threshold_db == DEFAULT_THRESHOLD_DB
        assert vad.hangover_frames == DEFAULT_HANGOVER_FRAMES
        assert vad._hangover_counter == 0

    def test_custom_parameters(self):
        vad = EnergyVAD(threshold_db=-30.0, hangover_frames=3)
        assert vad.threshold_db == -30.0
        assert vad.hangover_frames == 3

    def test_negative_hangover_raises(self):
        with pytest.raises(ValueError, match="hangover_frames"):
            EnergyVAD(hangover_frames=-1)

    def test_zero_hangover_allowed(self):
        vad = EnergyVAD(hangover_frames=0)
        assert vad.hangover_frames == 0


class TestEnergyVADSpeechDetection:
    def test_loud_sine_classified_as_speech(self):
        """A sine wave at -6 dB is well above the -40 dB threshold."""
        vad = EnergyVAD(threshold_db=-40.0, hangover_frames=0)
        result = vad.process(_sine(amp=0.5))
        assert result.is_speech is True

    def test_silence_classified_as_non_speech(self):
        """All-zero audio is far below any reasonable threshold."""
        vad = EnergyVAD(threshold_db=-40.0, hangover_frames=0)
        result = vad.process(_silence())
        assert result.is_speech is False

    def test_very_quiet_classified_as_non_speech(self):
        """Amplitude 1e-6 → ≈ -120 dB, well below -40 dB threshold."""
        vad = EnergyVAD(threshold_db=-40.0, hangover_frames=0)
        result = vad.process(_very_quiet())
        assert result.is_speech is False

    def test_above_custom_threshold(self):
        """Sine wave at amp=0.01 → ≈ -40 dB; must be speech above threshold=-50."""
        vad = EnergyVAD(threshold_db=-50.0, hangover_frames=0)
        result = vad.process(_sine(amp=0.01))
        assert result.is_speech is True

    def test_below_custom_threshold(self):
        """Sine wave at amp=0.001 → ≈ -60 dB; must be silence below threshold=-50."""
        vad = EnergyVAD(threshold_db=-50.0, hangover_frames=0)
        result = vad.process(_sine(amp=0.001))
        assert result.is_speech is False


class TestEnergyVADTimingPreservation:
    """VAD must NEVER modify audio samples."""

    def test_audio_unmodified_speech(self):
        vad = EnergyVAD()
        chunk = _sine()
        result = vad.process(chunk)
        np.testing.assert_array_equal(result.audio, chunk)

    def test_audio_unmodified_silence(self):
        vad = EnergyVAD()
        chunk = _silence()
        result = vad.process(chunk)
        np.testing.assert_array_equal(result.audio, chunk)

    def test_result_audio_is_same_object(self):
        """VADResult.audio must be the same object as the input chunk."""
        vad = EnergyVAD()
        chunk = _sine()
        result = vad.process(chunk)
        assert result.audio is chunk


class TestEnergyVADHangover:
    """Hangover prevents rapid toggling on brief pauses."""

    def test_hangover_holds_speech_label(self):
        """After speech, the next N non-speech frames still return is_speech=True."""
        hangover = 3
        vad = EnergyVAD(threshold_db=-40.0, hangover_frames=hangover)

        # First frame: speech
        r = vad.process(_sine())
        assert r.is_speech is True

        # Next `hangover` frames: silence — but label stays True
        for i in range(hangover):
            r = vad.process(_silence())
            assert r.is_speech is True, f"Frame {i+1} of hangover should still be speech"

    def test_hangover_expires_to_silence(self):
        """After hangover exhausted, next silence frame becomes non-speech."""
        hangover = 2
        vad = EnergyVAD(threshold_db=-40.0, hangover_frames=hangover)

        # Prime with speech
        vad.process(_sine())
        # Exhaust hangover
        for _ in range(hangover):
            vad.process(_silence())
        # Now should flip to silence
        r = vad.process(_silence())
        assert r.is_speech is False

    def test_hangover_reloads_on_new_speech(self):
        """New speech mid-hangover reloads the counter."""
        hangover = 4
        vad = EnergyVAD(threshold_db=-40.0, hangover_frames=hangover)

        vad.process(_sine())         # speech → counter=4
        vad.process(_silence())      # silence → counter=3
        vad.process(_silence())      # silence → counter=2
        vad.process(_sine())         # speech again → counter reloads to 4

        # Should have 4 more silence frames before flipping
        for i in range(hangover):
            r = vad.process(_silence())
            assert r.is_speech is True, f"After reload, frame {i+1} should still be speech"

        # Now should flip
        r = vad.process(_silence())
        assert r.is_speech is False

    def test_zero_hangover_flips_immediately(self):
        """With hangover_frames=0, silence immediately follows speech."""
        vad = EnergyVAD(threshold_db=-40.0, hangover_frames=0)
        vad.process(_sine())         # speech
        r = vad.process(_silence())  # should flip immediately
        assert r.is_speech is False


class TestEnergyVADReset:
    def test_reset_clears_hangover_counter(self):
        """After reset(), the hangover counter is back to 0."""
        vad = EnergyVAD(threshold_db=-40.0, hangover_frames=5)
        vad.process(_sine())         # loads counter to 5
        assert vad._hangover_counter == 5
        vad.reset()
        assert vad._hangover_counter == 0

    def test_reset_causes_silence_after_speech(self):
        """After reset(), silence frames are immediately non-speech."""
        vad = EnergyVAD(threshold_db=-40.0, hangover_frames=5)
        vad.process(_sine())   # prime hangover
        vad.reset()
        r = vad.process(_silence())
        assert r.is_speech is False

    def test_reset_on_fresh_vad_is_safe(self):
        vad = EnergyVAD()
        vad.reset()   # must not raise
        assert vad._hangover_counter == 0

    def test_double_reset_is_safe(self):
        vad = EnergyVAD()
        vad.process(_sine())
        vad.reset()
        vad.reset()
        assert vad._hangover_counter == 0


class TestEnergyVADEnergyDb:
    def test_energy_db_correct_for_known_sine(self):
        """Sine wave with amp=0.5 has RMS = 0.5/√2 ≈ 0.3536, ≈ -9.03 dB."""
        vad = EnergyVAD()
        result = vad.process(_sine(amp=0.5, n=16_000))  # 1 s for accuracy
        expected_db = 20.0 * np.log10(0.5 / np.sqrt(2.0))
        assert abs(result.energy_db - expected_db) < 0.5   # ±0.5 dB tolerance

    def test_energy_db_floor_for_silence(self):
        """Silence returns the floor value (-96 dB), not -inf."""
        vad = EnergyVAD()
        result = vad.process(_silence())
        assert result.energy_db <= -90.0

    def test_energy_db_is_float(self):
        vad = EnergyVAD()
        result = vad.process(_sine())
        assert isinstance(result.energy_db, float)


class TestEnergyVADEdgeCases:
    def test_2d_input_raises(self):
        vad = EnergyVAD()
        with pytest.raises(ValueError, match="1-D"):
            vad.process(np.zeros((2, CHUNK_SAMPLES), dtype=np.float32))

    def test_empty_chunk(self):
        """Empty chunk should return is_speech=False without raising."""
        vad = EnergyVAD(hangover_frames=0)
        result = vad.process(np.zeros(0, dtype=np.float32))
        assert result.is_speech is False
        assert result.energy_db <= -90.0


class TestRmsDb:
    """Unit tests for the internal _rms_db helper."""

    def test_full_scale_sine(self):
        """RMS of full-scale sine = 1/√2 → 0 dB in dB-RMS, but we use peak=1 so ≈ -3 dB."""
        audio = np.sin(np.linspace(0, 2 * np.pi, 16000)).astype(np.float32)
        db = _rms_db(audio)
        expected = 20.0 * np.log10(1.0 / np.sqrt(2.0))
        assert abs(db - expected) < 0.1

    def test_empty_returns_floor(self):
        assert _rms_db(np.zeros(0, dtype=np.float32)) <= -90.0

    def test_zero_signal_returns_floor(self):
        assert _rms_db(np.zeros(100, dtype=np.float32)) <= -90.0


# ===========================================================================
# NoiseReduceNoiseSuppressor
# ===========================================================================


class TestNoiseReduceNoiseSuppressor:
    def test_output_shape_matches_input(self):
        """Output must have the same number of samples as the input."""
        ns = NoiseReduceNoiseSuppressor()
        audio = _sine(n=CHUNK_SAMPLES * 4)
        result = ns.process(audio)
        assert result.shape == audio.shape

    def test_output_dtype_is_float32(self):
        ns = NoiseReduceNoiseSuppressor()
        audio = _sine(n=CHUNK_SAMPLES * 4)
        result = ns.process(audio)
        assert result.dtype == np.float32

    def test_short_signal_returned_unchanged(self):
        """Arrays shorter than 256 samples are returned as-is."""
        ns = NoiseReduceNoiseSuppressor()
        short = _sine(n=100)
        result = ns.process(short)
        np.testing.assert_array_equal(result, short)

    def test_2d_input_raises(self):
        ns = NoiseReduceNoiseSuppressor()
        with pytest.raises(ValueError, match="1-D"):
            ns.process(np.zeros((2, 1280), dtype=np.float32))

    def test_reset_is_safe(self):
        ns = NoiseReduceNoiseSuppressor()
        ns.reset()   # must not raise

    def test_silence_returns_near_silence(self):
        """Suppressing silence should produce near-silence output."""
        ns = NoiseReduceNoiseSuppressor(stationary=True)
        audio = _silence(n=CHUNK_SAMPLES * 4)
        result = ns.process(audio)
        assert result.dtype == np.float32
        assert result.shape == audio.shape
        # Silence in → should be very quiet out
        assert np.max(np.abs(result)) < 0.1

    def test_is_noise_suppressor_interface(self):
        assert isinstance(NoiseReduceNoiseSuppressor(), NoiseSuppressor)


# ===========================================================================
# PassthroughAEC
# ===========================================================================


class TestPassthroughAEC:
    def test_returns_copy_of_near_end(self):
        """Output is equal to near_end but is a distinct object."""
        aec = PassthroughAEC()
        near = _sine()
        result = aec.process(near)
        np.testing.assert_array_equal(result, near)
        assert result is not near   # must be a copy

    def test_dtype_preserved(self):
        aec = PassthroughAEC()
        near = _sine().astype(np.float32)
        result = aec.process(near)
        assert result.dtype == np.float32

    def test_no_far_end_no_warning(self):
        """Calling without far_end should not emit any warning."""
        aec = PassthroughAEC()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            aec.process(_sine())   # must not raise a warning

    def test_far_end_triggers_warning(self):
        """Providing far_end to PassthroughAEC must emit a UserWarning."""
        aec = PassthroughAEC(warn_on_far_end=True)
        with pytest.warns(UserWarning, match="PassthroughAEC received a far_end"):
            aec.process(_sine(), far_end=_sine())

    def test_far_end_warning_suppressed_when_disabled(self):
        """warn_on_far_end=False should suppress the warning."""
        aec = PassthroughAEC(warn_on_far_end=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            aec.process(_sine(), far_end=_sine())   # must not raise

    def test_2d_near_end_raises(self):
        aec = PassthroughAEC()
        with pytest.raises(ValueError, match="1-D"):
            aec.process(np.zeros((2, 1280), dtype=np.float32))

    def test_reset_is_safe(self):
        aec = PassthroughAEC()
        aec.reset()

    def test_is_aec_interface(self):
        assert isinstance(PassthroughAEC(), AcousticEchoCanceller)


# ===========================================================================
# Reset safety across all components
# ===========================================================================


class TestResetSafetyAllComponents:
    def test_energy_vad_reset_before_any_call(self):
        EnergyVAD().reset()

    def test_noisereduce_reset_before_any_call(self):
        NoiseReduceNoiseSuppressor().reset()

    def test_passthrough_aec_reset_before_any_call(self):
        PassthroughAEC().reset()

    def test_all_resets_after_use(self):
        vad = EnergyVAD()
        ns = NoiseReduceNoiseSuppressor()
        aec = PassthroughAEC()

        vad.process(_sine())
        aec.process(_sine())

        vad.reset()
        ns.reset()
        aec.reset()   # all must be safe


# ===========================================================================
# Pipeline decoupling — any compliant backend is accepted
# ===========================================================================


class TestPipelineDecoupling:
    """Demonstrate that the pipeline accepts any compliant backend."""

    def _run_vad_pipeline(self, vad: VoiceActivityDetector, chunk: np.ndarray) -> bool:
        assert isinstance(vad, VoiceActivityDetector)
        result = vad.process(chunk)
        assert isinstance(result, VADResult)
        assert result.audio is chunk
        return result.is_speech

    def test_energy_vad_accepted_by_pipeline(self):
        vad = EnergyVAD()
        self._run_vad_pipeline(vad, _sine())

    def test_custom_vad_accepted_by_pipeline(self):
        """A fully custom VAD implementation is accepted without pipeline changes."""
        class _AlwaysSpeechVAD(VoiceActivityDetector):
            def process(self, chunk):
                return VADResult(audio=chunk, is_speech=True, energy_db=0.0)
            def reset(self): pass

        vad = _AlwaysSpeechVAD()
        is_speech = self._run_vad_pipeline(vad, _silence())
        assert is_speech is True   # custom implementation overrides energy logic

    def test_swapping_vad_does_not_change_pipeline_function(self):
        """Two different VAD backends produce valid VADResult from the same function."""
        chunk = _sine()

        class _AlwaysSilenceVAD(VoiceActivityDetector):
            def process(self, c): return VADResult(audio=c, is_speech=False, energy_db=-96.0)
            def reset(self): pass

        for vad in [EnergyVAD(), _AlwaysSilenceVAD()]:
            result = self._run_vad_pipeline(vad, chunk)
            assert isinstance(result, bool)
