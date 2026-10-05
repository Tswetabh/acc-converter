"""
tests.unit.test_vocoder_synthesis
=================================
Unit tests for VocoderSynthesizer (src/accent_converter/models/backends/synthesis/vocoder.py).
"""

from unittest.mock import MagicMock
import numpy as np
import pytest
import torch

from accent_converter.models.backends.synthesis.acoustic_decoder import AcousticDecoder
from accent_converter.models.backends.synthesis.vocoder import VocoderSynthesizer


class TestVocoderSynthesizerUnit:
    """Unit tests for VocoderSynthesizer with mocked components (no network / GPU required)."""

    def test_synthesizer_init_state(self):
        synth = VocoderSynthesizer()
        assert not synth._loaded
        assert synth.BACKEND_NAME == "vocoder"

    def test_synthesize_before_load_raises_runtime_error(self):
        synth = VocoderSynthesizer()
        with pytest.raises(RuntimeError, match="called before load"):
            synth.synthesize(np.zeros((10, 768), dtype=np.float32), np.zeros(192, dtype=np.float32))

    def test_load_stub_mode(self):
        synth = VocoderSynthesizer()
        synth.load(checkpoint_path="stub")
        assert synth._loaded
        assert synth._is_stub

        # Returns matching silence
        out = synth.synthesize(np.zeros((20, 768), dtype=np.float32), np.zeros(192, dtype=np.float32))
        assert out.shape == (20 * 320,)
        assert out.dtype == np.float32
        assert (out == 0.0).all()

    def test_load_nonexistent_checkpoint_raises_file_not_found(self):
        synth = VocoderSynthesizer()
        with pytest.raises(FileNotFoundError, match="not found"):
            synth.load(checkpoint_path="nonexistent_checkpoint.pt")

    def test_load_invalid_device_raises_value_error(self, tmp_path):
        dummy_ckpt = tmp_path / "model.pt"
        torch.save({"acoustic_decoder": AcousticDecoder().state_dict()}, dummy_ckpt)

        synth = VocoderSynthesizer()
        with pytest.raises(ValueError, match="Invalid device"):
            synth.load(checkpoint_path=str(dummy_ckpt), device="invalid_device_name")

    def test_synthesize_with_mocked_vocoder(self, tmp_path):
        dummy_ckpt = tmp_path / "decoder.pt"
        decoder = AcousticDecoder(hidden_dim=64, n_conv_blocks=2)
        torch.save({"acoustic_decoder": decoder.state_dict()}, dummy_ckpt)

        synth = VocoderSynthesizer()
        # Mock HIFIGAN to avoid network download in unit test
        mock_vocoder = MagicMock()
        # (B, 80, T_mel) -> (B, 1, T_mel * 256)
        mock_vocoder.decode_batch.side_effect = lambda mel: torch.zeros(
            (mel.shape[0], 1, mel.shape[2] * 256), dtype=torch.float32
        )

        # Manually wire loaded state for fast unit test
        synth._acoustic_decoder = decoder.eval()
        synth._vocoder = mock_vocoder
        synth._device = "cpu"
        synth._is_stub = False
        synth._loaded = True

        rep = np.random.randn(40, 768).astype(np.float32)
        emb = np.random.randn(192).astype(np.float32)

        wav = synth.synthesize(rep, emb)
        assert isinstance(wav, np.ndarray)
        assert wav.ndim == 1
        assert wav.dtype == np.float32
        # 40 frames @ 50 Hz -> round(40*1.25) = 50 mel frames -> 50 * 256 = 12800 samples
        assert len(wav) == 50 * 256

    def test_synthesize_empty_input(self):
        synth = VocoderSynthesizer()
        synth.load(checkpoint_path="stub")
        wav = synth.synthesize(np.empty((0, 768), dtype=np.float32), np.zeros(192, dtype=np.float32))
        assert len(wav) == 0

    def test_synthesize_validation_bad_types(self):
        synth = VocoderSynthesizer()
        synth.load(checkpoint_path="stub")

        with pytest.raises(TypeError, match="converted_rep must be a numpy ndarray"):
            synth.synthesize([[1.0] * 768], np.zeros(192, dtype=np.float32))  # type: ignore

        with pytest.raises(TypeError, match="speaker_embedding must be a numpy ndarray"):
            synth.synthesize(np.zeros((10, 768), dtype=np.float32), [0.0] * 192)  # type: ignore

    def test_synthesize_validation_bad_dimensions(self):
        synth = VocoderSynthesizer()
        synth.load(checkpoint_path="stub")

        with pytest.raises(ValueError, match="speaker_embedding must be 1-D with shape"):
            synth.synthesize(np.zeros((10, 768), dtype=np.float32), np.zeros((1, 192), dtype=np.float32))

        with pytest.raises(ValueError, match="converted_rep must be 2-D"):
            synth.synthesize(np.zeros((10, 768, 1), dtype=np.float32), np.zeros(192, dtype=np.float32))

    def test_reset_is_safe(self):
        synth = VocoderSynthesizer()
        synth.reset()
        synth.load(checkpoint_path="stub")
        synth.reset()
