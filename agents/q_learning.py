import numpy as np
import random
from typing import Tuple, Sequence
from maze.basic_maze import Action

class QLearningAgent:
    def __init__(
        self, 
        maze_shape: Tuple[int, int], 
        action_Space: Sequence[Action], 
        alpha: float = 0.1, 
        gamma: float = 0.99, 
        epsilon: float = 0.2
    ) -> None:
        
        self.q_table = np.zeros((*maze_shape, len(action_Space)))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_space = list(action_Space)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.action_space)
        else:
            row, col = state
            best_action_index = np.argmax(self.q_table[row, col])
            return Action(best_action_index)

    def learn(
        self, 
        state: Tuple[int, int],
        action: Action, 
        reward: float, 
        next_state: Tuple[int, int]
    ) -> None:
        row, col = state
        next_row, next_col = next_state
        action_id = action.value

        td_target = reward + self.gamma * np.max(self.q_table[next_row, next_col])
        td_error = td_target - self.q_table[row, col, action_id]
        self.q_table[row, col, action_id] += self.alpha * td_error
