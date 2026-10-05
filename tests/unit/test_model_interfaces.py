"""
tests/unit/test_model_interfaces.py
=====================================
Interface-level tests for accent_converter.models

These tests verify that:

1. All four ABCs enforce the interface contract (cannot instantiate without
   implementing all abstract methods).
2. All four factory functions return the correct interface type.
3. The pipeline can accept different synthesis backends without changing any
   pipeline code (demonstrated with dummy in-test backends).
4. reset() is always safe to call, even before load().
5. The factory raises clear errors for unknown or misconfigured backends.
6. Concrete stubs correctly implement the interface (isinstance checks).
7. Backend swapping via config changes (vocoder ↔ tvtsyn) produces different
   instance types but identical interface compliance.

No ML models are downloaded or loaded.
HubertContentEncoder.load() is implemented but requires a network/local
checkpoint — it must NOT be called in unit tests (use integration tests).
The unloaded-state RuntimeError from encode() is verified instead.
"""

from __future__ import annotations

import numpy as np
import pytest

from accent_converter.models.interfaces import (
    ContentEncoder,
    SpeakerEncoder,
    AccentTranslator,
    SpeechSynthesizer,
)
from accent_converter.models.factory import (
    build_synthesizer,
    build_content_encoder,
    build_speaker_encoder,
    build_accent_translator,
)
from accent_converter.models.backends.synthesis.vocoder import VocoderSynthesizer
from accent_converter.models.backends.synthesis.tvtsyn import TVTSynSynthesizer
from accent_converter.models.backends.content.hubert import HubertContentEncoder
from accent_converter.models.backends.speaker.ecapa import EcapaSpeakerEncoder
from accent_converter.models.backends.accent.naive import NaiveAccentTranslator


# ===========================================================================
# Dummy in-test implementations
# (used to prove the pipeline works with ANY compliant backend)
# ===========================================================================


class _DummySynthesizer(SpeechSynthesizer):
    """Minimal compliant SpeechSynthesizer for pipeline decoupling tests."""

    BACKEND_NAME = "dummy"

    def __init__(self):
        self._reset_count = 0
        self._load_path = None

    def load(self, checkpoint_path: str) -> None:
        self._load_path = checkpoint_path

    def synthesize(
        self, converted_rep: np.ndarray, speaker_embedding: np.ndarray
    ) -> np.ndarray:
        # Returns silence of proportional length to the input
        n = max(1, len(converted_rep) * 2)
        return np.zeros(n, dtype=np.float32)

    def reset(self) -> None:
        self._reset_count += 1


class _DummyContentEncoder(ContentEncoder):
    BACKEND_NAME = "dummy"

    def load(self, checkpoint_path: str) -> None:
        pass

    def encode(self, audio: np.ndarray) -> np.ndarray:
        # Return a fixed-size content representation
        return np.zeros(256, dtype=np.float32)

    def reset(self) -> None:
        pass

    @property
    def frame_rate_hz(self) -> int:
        return 50

    @property
    def output_dim(self) -> int:
        return 256

    @property
    def lookahead_ms(self) -> int:
        return 0   # strictly causal dummy


class _DummySpeakerEncoder(SpeakerEncoder):
    BACKEND_NAME = "dummy"

    def load(self, checkpoint_path: str) -> None:
        pass

    def encode(self, reference_audio: np.ndarray) -> np.ndarray:
        return np.zeros(192, dtype=np.float32)

    def reset(self) -> None:
        pass


class _DummyAccentTranslator(AccentTranslator):
    BACKEND_NAME = "dummy"

    def load(self, checkpoint_path: str) -> None:
        pass

    def translate(
        self,
        content_rep: np.ndarray,
        target_accent: str,
    ) -> np.ndarray:
        return content_rep.copy()

    def reset(self) -> None:
        pass


# ===========================================================================
# A minimal "pipeline" function that only depends on the interface
# ===========================================================================


def _run_synthesis_pipeline(
    synthesizer: SpeechSynthesizer,
    converted_rep: np.ndarray,
    speaker_embedding: np.ndarray,
) -> np.ndarray:
    """Minimal synthesis pipeline function.

    Deliberately accepts only the SpeechSynthesizer ABC — not any concrete
    type.  This is the pattern the real pipeline must follow.
    """
    assert isinstance(synthesizer, SpeechSynthesizer)
    waveform = synthesizer.synthesize(converted_rep, speaker_embedding)
    assert isinstance(waveform, np.ndarray)
    return waveform


# ===========================================================================
# ABC enforcement
# ===========================================================================


class TestABCEnforcement:
    """Verify that instantiating an incomplete ABC raises TypeError."""

    def test_speech_synthesizer_cannot_be_instantiated_directly(self):
        with pytest.raises(TypeError):
            SpeechSynthesizer()  # type: ignore[abstract]

    def test_content_encoder_cannot_be_instantiated_directly(self):
        with pytest.raises(TypeError):
            ContentEncoder()  # type: ignore[abstract]

    def test_speaker_encoder_cannot_be_instantiated_directly(self):
        with pytest.raises(TypeError):
            SpeakerEncoder()  # type: ignore[abstract]

    def test_accent_translator_cannot_be_instantiated_directly(self):
        with pytest.raises(TypeError):
            AccentTranslator()  # type: ignore[abstract]

    def test_partial_implementation_cannot_be_instantiated(self):
        """A class that only implements some methods still raises TypeError."""
        class _Partial(SpeechSynthesizer):
            def load(self, path): pass
            def synthesize(self, r, s): return np.zeros(0)
            # reset() not implemented

        with pytest.raises(TypeError):
            _Partial()  # type: ignore[abstract]

    def test_content_encoder_missing_metadata_properties_raises(self):
        """ContentEncoder subclass missing the metadata properties cannot be
        instantiated, enforcing the interface contract."""
        class _MissingProps(ContentEncoder):
            def load(self, p): pass
            def encode(self, a): return np.zeros(10)
            def reset(self): pass
            # frame_rate_hz, output_dim, lookahead_ms NOT implemented

        with pytest.raises(TypeError):
            _MissingProps()  # type: ignore[abstract]


# ===========================================================================
# Pipeline decoupling — backend swapping without pipeline changes
# ===========================================================================


class TestPipelineDecoupling:
    """The pipeline function accepts any SpeechSynthesizer; no code changes needed."""

    REP = np.zeros(64, dtype=np.float32)
    EMB = np.zeros(192, dtype=np.float32)

    def test_dummy_backend_accepted_by_pipeline(self):
        synth = _DummySynthesizer()
        waveform = _run_synthesis_pipeline(synth, self.REP, self.EMB)
        assert waveform.dtype == np.float32

    def test_vocoder_stub_is_accepted_by_pipeline_type_check(self):
        """VocoderSynthesizer satisfies the isinstance check in the pipeline."""
        synth = VocoderSynthesizer()
        # synthesize() raises NotImplementedError — that's expected.
        # The pipeline type-check (isinstance) must pass before the call.
        assert isinstance(synth, SpeechSynthesizer)

    def test_tvtsyn_stub_is_accepted_by_pipeline_type_check(self):
        synth = TVTSynSynthesizer()
        assert isinstance(synth, SpeechSynthesizer)

    def test_swapping_backends_does_not_change_pipeline_function(self):
        """Demonstrate backend swap: same _run_synthesis_pipeline, different backends."""
        dummy_a = _DummySynthesizer()
        dummy_b = _DummySynthesizer()  # Could be a different class in real use

        wf_a = _run_synthesis_pipeline(dummy_a, self.REP, self.EMB)
        wf_b = _run_synthesis_pipeline(dummy_b, self.REP, self.EMB)

        assert wf_a.dtype == wf_b.dtype   # same contract, same output format

    def test_two_different_dummy_synthesizers_both_satisfy_interface(self):
        """Two classes implementing SpeechSynthesizer are interchangeable."""
        class _SilenceSynth(SpeechSynthesizer):
            def load(self, p): pass
            def synthesize(self, r, s): return np.zeros(16, dtype=np.float32)
            def reset(self): pass

        class _NoiseSynth(SpeechSynthesizer):
            def load(self, p): pass
            def synthesize(self, r, s): return np.ones(16, dtype=np.float32) * 0.01
            def reset(self): pass

        for synth in [_SilenceSynth(), _NoiseSynth()]:
            wf = _run_synthesis_pipeline(synth, self.REP, self.EMB)
            assert wf.ndim == 1


# ===========================================================================
# Factory tests
# ===========================================================================


class TestFactory:
    """build_* functions return the correct interface type."""

    def test_build_vocoder_synthesizer(self):
        synth = build_synthesizer({"backend": "vocoder"})
        assert isinstance(synth, SpeechSynthesizer)
        assert isinstance(synth, VocoderSynthesizer)

    def test_build_tvtsyn_synthesizer(self):
        synth = build_synthesizer({"backend": "tvtsyn"})
        assert isinstance(synth, SpeechSynthesizer)
        assert isinstance(synth, TVTSynSynthesizer)

    def test_build_content_encoder(self):
        enc = build_content_encoder({"backend": "hubert"})
        assert isinstance(enc, ContentEncoder)
        assert isinstance(enc, HubertContentEncoder)

    def test_build_speaker_encoder(self):
        enc = build_speaker_encoder({"backend": "ecapa"})
        assert isinstance(enc, SpeakerEncoder)
        assert isinstance(enc, EcapaSpeakerEncoder)

    def test_build_accent_translator(self):
        translator = build_accent_translator({"backend": "naive"})
        assert isinstance(translator, AccentTranslator)
        assert isinstance(translator, NaiveAccentTranslator)

    def test_factory_config_backend_switch_vocoder_to_tvtsyn(self):
        """Switching backend in config produces a different class, same interface."""
        synth_v = build_synthesizer({"backend": "vocoder"})
        synth_t = build_synthesizer({"backend": "tvtsyn"})
        assert type(synth_v) is not type(synth_t)
        assert isinstance(synth_v, SpeechSynthesizer)
        assert isinstance(synth_t, SpeechSynthesizer)

    def test_unknown_synthesis_backend_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown synthesis backend"):
            build_synthesizer({"backend": "does_not_exist"})

    def test_missing_backend_key_raises_value_error(self):
        with pytest.raises(ValueError, match="missing required key 'backend'"):
            build_synthesizer({"checkpoint": "path/to/something"})

    def test_unknown_content_encoder_raises(self):
        with pytest.raises(KeyError):
            build_content_encoder({"backend": "wavlm"})  # not yet registered

    def test_unknown_speaker_encoder_raises(self):
        with pytest.raises(KeyError):
            build_speaker_encoder({"backend": "titanet"})  # not yet registered

    def test_unknown_accent_translator_raises(self):
        with pytest.raises(KeyError):
            build_accent_translator({"backend": "seq2seq"})  # not yet registered


# ===========================================================================
# Concrete stubs — interface compliance
# ===========================================================================


class TestConcreteStubs:
    """Verify concrete stubs implement the full interface correctly."""

    def test_vocoder_synthesizer_is_speech_synthesizer(self):
        assert issubclass(VocoderSynthesizer, SpeechSynthesizer)

    def test_tvtsyn_synthesizer_is_speech_synthesizer(self):
        assert issubclass(TVTSynSynthesizer, SpeechSynthesizer)

    def test_hubert_is_content_encoder(self):
        assert issubclass(HubertContentEncoder, ContentEncoder)

    def test_ecapa_is_speaker_encoder(self):
        assert issubclass(EcapaSpeakerEncoder, SpeakerEncoder)

    def test_naive_is_accent_translator(self):
        assert issubclass(NaiveAccentTranslator, AccentTranslator)

    def test_vocoder_load_raises_file_not_found_with_nonexistent_checkpoint(self):
        synth = VocoderSynthesizer()
        with pytest.raises(FileNotFoundError):
            synth.load("dummy.ckpt")

    def test_vocoder_synthesize_raises_runtime_error_before_load(self):
        synth = VocoderSynthesizer()
        with pytest.raises(RuntimeError, match="called before load"):
            synth.synthesize(np.zeros((10, 768), dtype=np.float32), np.zeros(192, dtype=np.float32))

    def test_vocoder_load_stub_mode_and_synthesize(self):
        synth = VocoderSynthesizer()
        synth.load()  # checkpoint_path=None -> stub mode
        rep = np.zeros((15, 768), dtype=np.float32)
        emb = np.zeros(192, dtype=np.float32)
        waveform = synth.synthesize(rep, emb)
        assert waveform.ndim == 1
        assert waveform.dtype == np.float32
        assert len(waveform) == 15 * 320
        np.testing.assert_array_equal(waveform, np.zeros(15 * 320, dtype=np.float32))

    def test_vocoder_synthesize_input_validation(self):
        synth = VocoderSynthesizer()
        synth.load()
        # Bad speaker embedding shape
        with pytest.raises(ValueError, match="speaker_embedding"):
            synth.synthesize(np.zeros((10, 768)), np.zeros(100))
        # Bad converted_rep shape (1-D instead of 2-D)
        with pytest.raises(ValueError, match="converted_rep"):
            synth.synthesize(np.zeros(768), np.zeros(192))

    def test_tvtsyn_load_raises_not_implemented(self):
        synth = TVTSynSynthesizer()
        with pytest.raises(NotImplementedError):
            synth.load("dummy.ckpt")

    def test_tvtsyn_synthesize_raises_not_implemented(self):
        synth = TVTSynSynthesizer()
        with pytest.raises(NotImplementedError):
            synth.synthesize(np.zeros(64), np.zeros(192))

    def test_hubert_encode_raises_runtime_error_before_load(self):
        """encode() must raise RuntimeError if load() has not been called.

        This test does NOT call load() (no network/model download required).
        It verifies the unloaded-state guard that protects the inference path.
        """
        enc = HubertContentEncoder()
        audio = np.zeros(1600, dtype=np.float32)
        with pytest.raises(RuntimeError, match="called before load"):
            enc.encode(audio)

    def test_ecapa_encode_raises_runtime_error_before_load(self):
        enc = EcapaSpeakerEncoder()
        audio = np.zeros(1600, dtype=np.float32)
        with pytest.raises(RuntimeError, match="called before load"):
            enc.encode(audio)

    def test_naive_translator_load_is_safe(self):
        """NaiveAccentTranslator.load() is a no-op (no real weights)."""
        t = NaiveAccentTranslator()
        t.load("ignored")   # must not raise
        assert t._loaded is True

    def test_naive_translator_translate_is_passthrough(self):
        """NaiveAccentTranslator returns a copy of the content rep unchanged."""
        t = NaiveAccentTranslator()
        content = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = t.translate(content, target_accent="us")
        np.testing.assert_array_equal(result, content)
        assert result is not content   # must be a copy


# ===========================================================================
# reset() safety
# ===========================================================================


class TestResetSafety:
    """reset() must be safe to call before load() and multiple times."""

    def test_vocoder_reset_before_load_is_safe(self):
        VocoderSynthesizer().reset()   # must not raise

    def test_tvtsyn_reset_before_load_is_safe(self):
        TVTSynSynthesizer().reset()

    def test_hubert_reset_before_load_is_safe(self):
        HubertContentEncoder().reset()

    def test_ecapa_reset_is_safe(self):
        EcapaSpeakerEncoder().reset()

    def test_naive_reset_is_safe(self):
        NaiveAccentTranslator().reset()

    def test_dummy_reset_increments_counter(self):
        synth = _DummySynthesizer()
        assert synth._reset_count == 0
        synth.reset()
        synth.reset()
        assert synth._reset_count == 2


# ===========================================================================
# ContentEncoder metadata properties
# ===========================================================================


class TestContentEncoderMetadata:
    """Verify metadata property contract on ContentEncoder implementations."""

    def test_hubert_frame_rate_hz_is_50(self):
        """HuBERT CNN stride → 50 Hz (20 ms / frame)."""
        enc = HubertContentEncoder()
        assert enc.frame_rate_hz == 50

    def test_hubert_output_dim_is_768(self):
        """HuBERT Base hidden size is 768."""
        enc = HubertContentEncoder()
        assert enc.output_dim == 768

    def test_hubert_lookahead_ms_is_minus_one(self):
        """HuBERT is non-causal: lookahead_ms sentinel must be -1."""
        enc = HubertContentEncoder()
        assert enc.lookahead_ms == -1

    def test_hubert_is_not_causal(self):
        """Confirm the is-causal check pattern: lookahead_ms >= 0."""
        enc = HubertContentEncoder()
        is_safe_for_streaming = enc.lookahead_ms >= 0
        assert is_safe_for_streaming is False, (
            "HuBERT must not be flagged as streaming-safe"
        )

    def test_dummy_encoder_frame_rate(self):
        enc = _DummyContentEncoder()
        assert enc.frame_rate_hz == 50

    def test_dummy_encoder_output_dim(self):
        enc = _DummyContentEncoder()
        assert enc.output_dim == 256

    def test_dummy_encoder_lookahead_causal(self):
        """Dummy encoder declares strictly causal (lookahead_ms == 0)."""
        enc = _DummyContentEncoder()
        assert enc.lookahead_ms == 0
        assert enc.lookahead_ms >= 0  # streaming-safe check passes

    def test_metadata_properties_are_ints(self):
        enc = HubertContentEncoder()
        assert isinstance(enc.frame_rate_hz, int)
        assert isinstance(enc.output_dim, int)
        assert isinstance(enc.lookahead_ms, int)

    def test_custom_streaming_encoder_accepted_by_pipeline(self):
        """Any ContentEncoder with lookahead_ms >= 0 is streaming-safe."""
        class _CausalEncoder(ContentEncoder):
            def load(self, p): pass
            def encode(self, a): return np.zeros((4, 128), dtype=np.float32)
            def reset(self): pass

            @property
            def frame_rate_hz(self): return 100   # 10 ms / frame

            @property
            def output_dim(self): return 128

            @property
            def lookahead_ms(self): return 40   # bounded look-ahead

        enc = _CausalEncoder()
        assert isinstance(enc, ContentEncoder)
        assert enc.lookahead_ms >= 0  # streaming-safe
        assert enc.frame_rate_hz == 100
        assert enc.output_dim == 128
