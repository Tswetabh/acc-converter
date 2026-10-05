"""
tests/integration/test_synthesis_integration.py
===============================================
Integration test for real neural speech synthesis:
AcousticDecoder + 16 kHz HiFi-GAN Vocoder (speechbrain/tts-hifigan-libritts-16kHz).

Marked with @pytest.mark.integration (skipped during unit test runs).
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pytest
import torch

from accent_converter.models.backends.synthesis.acoustic_decoder import AcousticDecoder
from accent_converter.models.backends.synthesis.vocoder import VocoderSynthesizer


@pytest.mark.integration
class TestSynthesisIntegration:
    """Tests real neural synthesis using HiFi-GAN and AcousticDecoder."""

    def test_vocoder_real_synthesis_pipeline(self, tmp_path: Path):
        # 1. Save initialized AcousticDecoder checkpoint
        ckpt_path = tmp_path / "acoustic_decoder.pt"
        decoder = AcousticDecoder()
        torch.save({"acoustic_decoder": decoder.state_dict()}, ckpt_path)

        # 2. Instantiate and load VocoderSynthesizer
        synth = VocoderSynthesizer()
        synth.load(
            checkpoint_path=str(ckpt_path),
            vocoder_source="models/hifigan_16k",
        )
        assert synth._loaded
        assert not synth._is_stub

        # 3. Synthesize from synthetic HuBERT (1.0s = 50 frames) and ECAPA (192,)
        rep = np.random.randn(50, 768).astype(np.float32)
        emb = np.random.randn(192).astype(np.float32)
        # Normalize embedding like real ECAPA
        emb = emb / (np.linalg.norm(emb) + 1e-8)

        wav = synth.synthesize(rep, emb)

        # 4. Verify waveform properties
        assert isinstance(wav, np.ndarray)
        assert wav.ndim == 1
        assert wav.dtype == np.float32
        assert len(wav) > 0
        # Should NOT be all zeros (neural network output)
        assert not np.allclose(wav, 0.0)
        # Duration should be close to 1.0s (around 16000-18432 samples including inference padding)
        assert 14000 <= len(wav) <= 20000
