import numpy as np
import random
from maze.basic_maze import Action

class QLearningAgent:
    def __init__(self, maze_shape, n_actions, alpha=0.1, gamma=0.99, epsilon=0.2):
        self.q_table = np.zeros((*maze_shape, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(list(Action))
        else:
            row, col = state
            best_action_index = np.argmax(self.q_table[row, col])
            return Action(best_action_index)

    def learn(self, state, action, reward, next_state):
        row, col = state
        next_row, next_col = next_state
        a = action.value

        td_target = reward + self.gamma * np.max(self.q_table[next_row, next_col])
        td_error = td_target - self.q_table[row, col, a]
        self.q_table[row, col, a] += self.alpha * td_error
