# models/backends/synthesis/__init__.py
from accent_converter.models.backends.synthesis.vocoder import VocoderSynthesizer
from accent_converter.models.backends.synthesis.tvtsyn import TVTSynSynthesizer

__all__ = ["VocoderSynthesizer", "TVTSynSynthesizer"]
