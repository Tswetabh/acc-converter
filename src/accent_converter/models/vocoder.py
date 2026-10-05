# Causal Vocoder/Decoder
from .base import BaseModel

class CausalVocoder(BaseModel):
    def decode(self, representation):
        raise NotImplementedError("Vocoder decoding not implemented.")
