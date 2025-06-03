from dataclasses import dataclass, asdict
import json

"""
Defines the Hyperparameters dataclass for configuring reinforcement learning experiments.

This class provides a convenient way to store, serialize, and load experiment settings
such as learning rate, discount factor, epsilon, and maze configuration.
"""

@dataclass
class Hyperparameters:
    episodes: int = 1000
    max_steps: int = 1000
    learning_rate: float = 0.1
    discount_factor: float = 0.99
    decay_epsilon: bool = True
    alpha: float = 0.1
    epsilon: float = 0.2
    gamma: float = 0.99
    maze_name: str = "Test"
    start_cell_idx: int = 0

    def to_json(
            self, 
            filepath: str
        ) -> None:
        """
        Save the hyperparameters to a JSON file.

        Args:
            filepath: Path to the output JSON file.
        """
        with open(filepath, "w") as f:
            json.dump(asdict(self), f, indent=4)

    @staticmethod
    def from_json(
        filepath: str
    ) -> "Hyperparameters":
        """
        Load hyperparameters from a JSON file.

        Args:
            filepath: Path to the JSON file.

        Returns:
            An instance of Hyperparameters with values loaded from the file.
        """
        with open(filepath, "r") as f:
            data = json.load(f)
        return Hyperparameters(**data)
