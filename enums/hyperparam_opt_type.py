from enum import Enum

class HyperparamOptType(Enum):
    """
    Enum for different types of hyperparameter optimization methods.
    """
    GRID_SEARCH = "grid_search"
    EVOLUTIONARY_ALGORITHM = "evolutionary_algorithm"
    OPTUNA = "optuna"