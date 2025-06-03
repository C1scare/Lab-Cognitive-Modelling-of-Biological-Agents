import numpy as np
import random
from typing import Tuple, Sequence
from maze.basic_maze import Action
from agents.base_agent import BaseAgent
from training.hyperparameter import Hyperparameter

class QLearningAgent(BaseAgent):
    """
    A simple Q-learning agent for grid-based environments like BasicMaze.

    Attributes:
        q_table: Q-values for each state-action pair.
        alpha: Learning rate.
        gamma: Discount factor.
        epsilon: Exploration rate for ε-greedy policy.
        action_space: List of possible actions.
    """

    def __init__(
        self, 
        maze_shape: Tuple[int, int], 
        action_Space: Sequence[Action],
        hyperparameters: Hyperparameter = Hyperparameter(
            alpha=0.1, 
            gamma=0.99, 
            epsilon=0.2
        )
    ) -> None:
        """
        Initialize the Q-learning agent with an empty Q-table.

        Args:
            maze_shape: Dimensions of the maze grid.
            action_Space: List of possible actions (e.g., UP, DOWN, etc.).
            alpha: Learning rate for Q-value updates.
            gamma: Discount factor for future rewards.
            epsilon: Initial exploration rate.
        """
        self.q_table = np.zeros((*maze_shape, len(action_Space)))
        if (hyperparameters.alpha is None or
            hyperparameters.gamma is None or
            hyperparameters.epsilon is None):
            raise ValueError("Hyperparameters must be provided with valid values.")
        self.alpha = hyperparameters.alpha
        self.gamma = hyperparameters.gamma
        self.epsilon = hyperparameters.epsilon
        self.action_space = list(action_Space)

    def choose_action(self, state: Tuple[int, int]) -> Action:
        """
        Select an action using ε-greedy strategy.

        Args:
            state: Current state of the agent in the maze.

        Returns:
            An Action selected either randomly or greedily.
        """
        if random.random() < self.epsilon:
            return random.choice(self.action_space)
        row, col = state
        best_action_index = np.argmax(self.q_table[row, col])
        return Action(best_action_index)

    def learn(
        self, 
        state: Tuple[int, int],
        action: Action, 
        reward: float, 
        next_state: Tuple[int, int],
        done: bool = False
    ) -> None:
        """
        Update the Q-value for a given state-action pair using the Bellman equation.

        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Resulting state after taking the action.
        """
        row, col = state
        next_row, next_col = next_state
        action_id = action.value

        td_target = reward + self.gamma * np.max(self.q_table[next_row, next_col])
        td_error = td_target - self.q_table[row, col, action_id]
        self.q_table[row, col, action_id] += self.alpha * td_error

    def decay_epsilon(self, decay_rate: float = 0.99, min_epsilon: float = 0.01) -> None:
        """
        Reduce the exploration rate over time.

        Args:
            decay_rate: Multiplicative factor to reduce epsilon.
            min_epsilon: Lower bound for epsilon.
        """
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
