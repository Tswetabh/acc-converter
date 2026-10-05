# Streaming cache management
class AudioCache:
    def __init__(self):
        self.buffer = None

    def reset(self):
        self.buffer = None
