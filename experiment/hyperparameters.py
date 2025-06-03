from dataclasses import dataclass, asdict
import json


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

    def to_json(self, filepath: str) -> None:
        with open(filepath, "w") as f:
            json.dump(asdict(self), f, indent=4)

    @staticmethod
    def from_json(filepath: str) -> "Hyperparameters":
        with open(filepath, "r") as f:
            data = json.load(f)
        return Hyperparameters(**data)
