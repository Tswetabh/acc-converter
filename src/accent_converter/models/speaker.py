# Speaker Encoder
from .base import BaseModel

class SpeakerEncoder(BaseModel):
    def encode(self, reference_audio):
        raise NotImplementedError("Speaker encoding not implemented.")
