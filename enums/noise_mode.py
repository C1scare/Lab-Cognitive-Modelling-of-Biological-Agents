from enum import Enum

class NoiseMode(Enum):
    PERCEPTUAL = "Perceptual noise"
    NEURAL = "Neural noise"
    BOTH = "Both perceptual and neural noise"
    NONE = "No noise"
