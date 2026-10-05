# Content Encoder
from .base import BaseModel

class ContentEncoder(BaseModel):
    def encode(self, audio):
        raise NotImplementedError("Content encoding not implemented.")
