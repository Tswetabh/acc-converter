"""
tests.unit.test_acoustic_decoder
================================
Unit tests for the AcousticDecoder PyTorch module (src/accent_converter/models/backends/synthesis/acoustic_decoder.py).
"""

import pytest
import torch

from accent_converter.models.backends.synthesis.acoustic_decoder import AcousticDecoder, ConvBlock


class TestAcousticDecoder:
    """Unit tests for AcousticDecoder architecture and forward pass."""

    def test_conv_block_shape_and_residual(self):
        block = ConvBlock(in_channels=64, out_channels=64, kernel_size=5)
        x = torch.randn(2, 64, 50)
        out = block(x)
        assert out.shape == (2, 64, 50)

    def test_forward_output_shape_batched(self):
        model = AcousticDecoder(
            content_dim=768, speaker_dim=192, mel_dim=80, hidden_dim=128, n_conv_blocks=2
        )
        hubert = torch.randn(4, 50, 768)  # 50 frames @ 50 Hz = 1.0s
        speaker = torch.randn(4, 192)

        mel = model(hubert, speaker)
        # Expected frames: round(50 * 1.25) = 62 (banker's rounding) @ 62.5 Hz
        assert mel.shape == (4, 80, 62)
        assert mel.dtype == torch.float32

    def test_forward_output_shape_unbatched(self):
        model = AcousticDecoder(hidden_dim=128, n_conv_blocks=2)
        hubert = torch.randn(40, 768)
        speaker = torch.randn(192)

        mel = model(hubert, speaker)
        # Expected frames: round(40 * 1.25) = 50
        assert mel.shape == (1, 80, 50)

    def test_forward_explicit_target_len(self):
        model = AcousticDecoder(hidden_dim=128, n_conv_blocks=2)
        hubert = torch.randn(2, 40, 768)
        speaker = torch.randn(2, 192)

        mel = model(hubert, speaker, target_len=75)
        assert mel.shape == (2, 80, 75)

    def test_empty_content_sequence(self):
        model = AcousticDecoder(hidden_dim=128, n_conv_blocks=2)
        hubert = torch.randn(2, 0, 768)
        speaker = torch.randn(2, 192)

        mel = model(hubert, speaker)
        assert mel.shape == (2, 80, 0)

    def test_invalid_content_dim_raises_value_error(self):
        model = AcousticDecoder(content_dim=768, speaker_dim=192)
        hubert_bad = torch.randn(2, 30, 512)
        speaker = torch.randn(2, 192)

        with pytest.raises(ValueError, match="Expected content_dim=768"):
            model(hubert_bad, speaker)

    def test_invalid_speaker_dim_raises_value_error(self):
        model = AcousticDecoder(content_dim=768, speaker_dim=192)
        hubert = torch.randn(2, 30, 768)
        speaker_bad = torch.randn(2, 128)

        with pytest.raises(ValueError, match="Expected speaker_dim=192"):
            model(hubert, speaker_bad)

    def test_backward_pass_gradients_flow(self):
        model = AcousticDecoder(hidden_dim=64, n_conv_blocks=2)
        hubert = torch.randn(2, 20, 768, requires_grad=True)
        speaker = torch.randn(2, 192, requires_grad=True)

        mel = model(hubert, speaker, target_len=25)
        target = torch.randn_like(mel)
        loss = torch.nn.functional.l1_loss(mel, target)
        loss.backward()

        # Check gradients for model weights
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient"

    def test_eval_mode_deterministic(self):
        model = AcousticDecoder(hidden_dim=64, n_conv_blocks=2, dropout=0.5)
        model.eval()

        hubert = torch.randn(1, 30, 768)
        speaker = torch.randn(1, 192)

        with torch.no_grad():
            out1 = model(hubert, speaker)
            out2 = model(hubert, speaker)

        torch.testing.assert_close(out1, out2)
