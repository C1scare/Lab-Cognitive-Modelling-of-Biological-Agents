from enum import Enum

class NoiseMode(Enum):
    """
    Enum for different types of noise modes used in agents.
    """
    PERCEPTUAL = "Perceptual noise"
    NEURAL = "Neural noise"
    BOTH = "Both perceptual and neural noise"
    NONE = "No noise"
