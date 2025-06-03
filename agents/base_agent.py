from abc import ABC, abstractmethod
import pickle
from pathlib import Path
from typing import Tuple, Sequence
from maze.basic_maze import Action

class BaseAgent(ABC):
    """
    Abstract base class for agents in grid-based environments.
    """
    def __init__(
        self, 
        maze_shape: Tuple[int, int], 
        action_space: Sequence[Action], 
    ) -> None:
        """
        Initialize the agent.

        Args:
            maze_shape: Dimensions of the maze grid.
            action_space: List of possible actions (e.g., UP, DOWN, etc.).
        """
        self.action_space = list(action_space)

    
    @abstractmethod
    def choose_action(self, state: Tuple[int, int]) -> Action:
        """
        Select an action given the current state.

        Args:
            state: Current position of the agent in the maze.

        Returns:
            An action to take.
        """
        pass

    @abstractmethod
    def learn(
        self, 
        state: Tuple[int, int], 
        action: Action, 
        reward: float, 
        next_state: Tuple[int, int]
    ) -> None:
        """
        Update the agent's knowledge using the observed transition.

        Args:
            state: The previous state.
            action: The action taken.
            reward: The reward received.
            next_state: The resulting state.
        """
        pass

    def save_agent(self, filepath: str) -> None:
        """Save any agent object to a pickle file."""
        path: Path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_agent(filepath: str) -> "BaseAgent":
        """Load any agent object from a pickle file."""
        path = Path(filepath)
        with open(path.with_suffix(".pkl"), "rb") as f:
            return pickle.load(f)

    def decay_epsilon(self) -> None:
        """
        Decay the exploration rate (if applicable).
        This method can be overridden by subclasses if they implement epsilon decay.
        """
        pass