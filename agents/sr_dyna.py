import numpy as np
import random
from collections import deque
from typing import Tuple, Sequence

# Assuming the following classes are defined in these files
from maze.basic_maze import Action
from agents.base_agent import BaseAgent
from training.hyperparameter import Hyperparameter

class SRDynaAgent(BaseAgent):
    """
    An agent that implements the SR-Dyna algorithm as described in the provided paper.

    SR-Dyna learns a state-action successor representation (H-matrix) and a reward
    weight vector (w). Q-values are computed from the dot product of these two.
    It utilizes offline replay of experienced transitions to update the successor
    representation, allowing it to adapt to changes in the environment's dynamics.

    Attributes:
        maze_shape: The dimensions (rows, cols) of the grid environment.
        action_space: A sequence of possible actions.
        H: The state-action successor matrix.
        w: The reward weight vector.
        experience_buffer: A deque to store past transitions for replay.
        alpha_sr: Learning rate for the successor representation (H).
        alpha_w: Learning rate for the reward weights (w).
        gamma: Discount factor for future rewards.
        epsilon: Exploration rate for the ε-greedy policy.
        k: The number of transitions to replay from the buffer in each learning step.
        num_actions: The total number of possible actions.
        num_states: The total number of states in the maze.
        total_state_actions: The combined size of the state-action space.
    """

    def __init__(
        self,
        maze_shape: Tuple[int, int],
        action_space: Sequence[Action],
        hyperparameters: Hyperparameter = Hyperparameter(
            alpha_w=0.3, 
            alpha_sr=0.3,
            gamma=0.95,
            epsilon=0.1,
            k=10
        )
    ) -> None:
        """
        Initializes the SR-Dyna agent.

        Args:
            maze_shape: Dimensions of the maze grid.
            action_space: List of possible actions.
            hyperparameters: An object containing learning parameters like
                             alpha_w, alpha_sr, gamma, epsilon, and k.
        """
        if not all(hasattr(hyperparameters, attr) for attr in ['alpha_w', 'alpha_sr', 'gamma', 'epsilon', 'k']):
            raise AttributeError("Provided hyperparameters are missing one or more required attributes for SRDynaAgent: 'alpha_w', 'alpha_sr', 'k'.")

        self.maze_shape = maze_shape
        self.action_space = list(action_space)
        self.num_actions = len(self.action_space)
        self.num_states = maze_shape[0] * maze_shape[1]
        self.total_state_actions = self.num_states * self.num_actions

        # Initialize H as an identity matrix [cite: 520]
        self.H = np.identity(self.total_state_actions)
        # Initialize w as a zero vector [cite: 521]
        self.w = np.zeros(self.total_state_actions)

        self.gamma = hyperparameters.gamma
        self.epsilon = hyperparameters.epsilon
        self.alpha_sr = hyperparameters.alpha_sr
        self.alpha_w = hyperparameters.alpha_w
        self.k = hyperparameters.k
        
        self.experience_buffer = deque()
        
        self.seed = hyperparameters.random_seed
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

    def _map_state_action_to_index(self, state: Tuple[int, int], action: Action) -> int:
        """Helper to convert a (state, action) pair to a flat index."""
        row, col = state
        state_index = row * self.maze_shape[1] + col
        return state_index * self.num_actions + action.value

    def _get_q_values_for_state(self, state: Tuple[int, int]) -> np.ndarray:
        """
        Computes the Q-values for all actions in a given state.
        Q(s,a) is calculated as the dot product of the corresponding row in the 
        successor matrix H and the weight vector w. [cite: 310, 313]
        """
        q_values = np.zeros(self.num_actions)
        for action in self.action_space:
            sa_index = self._map_state_action_to_index(state, action)
            # Q(s,a) = H(sa, :) . w
            q_values[action.value] = self.H[sa_index, :] @ self.w
        return q_values

    def choose_action(self, state: Tuple[int, int]) -> Action:
        """
        Selects an action using an ε-greedy policy based on the current Q-values. [cite: 532]
        """
        if random.random() < self.epsilon:
            return random.choice(self.action_space)
        
        q_values = self._get_q_values_for_state(state)
        best_action_index = np.argmax(q_values)
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
        Updates the agent's knowledge, including online updates to H and w,
        storing the experience, and triggering offline replay to update H.
        """
        # --- 1. Store Experience ---
        self.experience_buffer.append((state, action, reward, next_state))

        # --- 2. Online Update (On-Policy) ---
        sa_idx = self._map_state_action_to_index(state, action)
        
        # To perform the SARSA-like update, we need the next action
        next_action = self.choose_action(next_state)
        s_prime_a_prime_idx = self._map_state_action_to_index(next_state, next_action)

        # Update weights w using TD-learning rule (Eq 15) [cite: 313]
        q_sa = self.H[sa_idx, :] @ self.w
        q_s_prime_a_prime = self.H[s_prime_a_prime_idx, :] @ self.w
        w_td_error = reward + self.gamma * q_s_prime_a_prime - q_sa
        self.w += self.alpha_w * w_td_error * self.H[sa_idx, :]

        # Update successor matrix H using TD-learning rule (Eq 17) [cite: 314]
        identity_row = np.zeros(self.total_state_actions)
        identity_row[sa_idx] = 1
        h_td_error = identity_row + self.gamma * self.H[s_prime_a_prime_idx, :] - self.H[sa_idx, :]
        self.H[sa_idx, :] += self.alpha_sr * h_td_error

        # --- 3. Offline Replay (Off-Policy Dyna Update) ---
        if len(self.experience_buffer) > self.k > 0:
            samples = random.choices(self.experience_buffer, k=self.k)
            for s_samp, a_samp, _, s_prime_samp in samples:
                self._replay_experience(s_samp, a_samp, s_prime_samp)

    def _replay_experience(self, state: Tuple[int, int], action: Action, next_state: Tuple[int, int]) -> None:
        """
        Performs an off-policy update on the H matrix for a single replayed transition.
        """
        sa_idx = self._map_state_action_to_index(state, action)

        # Find the best next action a* from the next state (off-policy) [cite: 318]
        q_values_next = self._get_q_values_for_state(next_state)
        best_next_action = Action(np.argmax(q_values_next))
        s_prime_a_star_idx = self._map_state_action_to_index(next_state, best_next_action)

        # Update H with the optimal next state-action (Eq 18) [cite: 318]
        identity_row = np.zeros(self.total_state_actions)
        identity_row[sa_idx] = 1.0
        h_td_error = identity_row + self.gamma * self.H[s_prime_a_star_idx, :] - self.H[sa_idx, :]
        self.H[sa_idx, :] += self.alpha_sr * h_td_error

    def decay_epsilon(self, decay_rate: float = 0.99, min_epsilon: float = 0.01) -> None:
        """Reduces the exploration rate over time."""
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)