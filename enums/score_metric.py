from enum import Enum

class ScoreMetric(Enum):
    """
    Enum for different types of scoring metrics used in model evaluation.
    """
    SUCCESS_RATE = "success_rate"
    AVERAGE_REWARD = "average_reward"
    MAX_REWARD = "max_reward"
    LEARNING_SPEED = "learning_speed"
    PATH_LENGTH = "best_path_length"