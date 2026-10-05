# Abstract base classes requiring reset() for state
class BaseModel:
    def reset(self):
        raise NotImplementedError("BaseModel.reset not implemented.")
