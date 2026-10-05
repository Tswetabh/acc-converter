# Accent Translator
from .base import BaseModel

class AccentTranslator(BaseModel):
    def translate(self, content_rep):
        raise NotImplementedError("Accent translation not implemented.")
