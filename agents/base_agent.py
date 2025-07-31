from abc import ABC, abstractmethod
from typing import Tuple, Sequence
from maze.basic_maze import Action

class BaseAgent(ABC):
    """
    Abstract base class for agents in grid-based environments.

    Provides the interface for action selection and learning.
    All agents must implement the choose_action and learn methods.
    """
    def __init__(
        self, 
        maze_shape: Tuple[int, int], 
        action_space: Sequence[Action], 
    ) -> None:
        """
        Initialize the agent.

        Args:
            maze_shape: Dimensions of the maze grid as (rows, columns).
            action_space: Sequence of possible actions ([Action.UP, Action.DOWN, Action.LEFT, Action.Right]).
        """
        self.action_space = list(action_space)

    
    @abstractmethod
    def choose_action(self, state: Tuple[int, int]) -> Action:
        """
        Select an action given the current state.

        Args:
            state: The agent's current position in the maze as (row, column).

        Returns:
            The chosen action.
        """
        pass

    @abstractmethod
    def learn(
        self, 
        state: Tuple[int, int], 
        action: Action, 
        reward: float, 
        next_state: Tuple[int, int],
        done: bool = False
    ) -> None:
        """
        Update the agent's knowledge based on the observed transition.

        Args:
            state: Previous state (row, column).
            action: Action taken.
            reward: Reward received after taking the action.
            next_state: Resulting state after the action.
            done: Whether the episode has ended.
        """
        pass