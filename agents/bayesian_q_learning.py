import numpy as np
import random
from typing import Tuple, Sequence
from maze.basic_maze import Action
from agents.base_agent import BaseAgent
from training.hyperparameter import Hyperparameter


class BayesianQLearningAgent(BaseAgent):
    """
    A Bayesian Q-learning agent for grid-based environments.

    This agent maintains a Gaussian distribution (mean and variance) for each Q-value,
    allowing for uncertainty-aware exploration using Thompson Sampling and Bayesian updates.

    Attributes:
        q_dist_table: 4D numpy array storing [mean, variance] for each (state, action) pair.
        alpha: Learning rate (from hyperparameters, may not be used directly).
        gamma: Discount factor for future rewards.
        epsilon: Exploration rate for ε-greedy policy.
        action_space: List of possible actions.
        tau_obs: Precision (inverse variance) of the Bellman target.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self, 
        maze_shape: Tuple[int, int], 
        action_space: Sequence[Action], 
        hyperparameters: Hyperparameter = Hyperparameter(
            gamma=0.99,
            epsilon=0.2,
            mu_init=0.0,
            sigma_sq_init=2.0,
            obs_noise_variance=0.1
        )
    ) -> None:
        """
        Initialize the Bayesian Q-learning agent.
        
        Args:
            maze_shape: Dimensions of the maze grid.
            action_space: List of possible actions (e.g., UP, DOWN, etc.).
            hyperparameters: Hyperparameter object with the following attributes:
                - gamma: Discount factor for future rewards.
                - epsilon: Initial exploration rate.
                - mu_init: Initial mean for the Gaussian Q-value distributions.
                - sigma_sq_init: Initial variance for the Gaussian Q-value distributions.
                - obs_noise_variance: Assumed variance of the Bellman target 'y'.
        Raises:
            ValueError: If any required hyperparameter is missing or invalid.
        """
        if(hyperparameters.gamma is None or
           hyperparameters.epsilon is None or
           hyperparameters.mu_init is None or
           hyperparameters.sigma_sq_init is None or
           hyperparameters.obs_noise_variance is None):
            raise ValueError("Hyperparameters must be provided with valid values.")

        self.q_dist_table = np.zeros((*maze_shape, len(action_space), 2))  # [mean, variance]
        self.q_dist_table[:, :, :, 0] = hyperparameters.mu_init  # Initialize means
        self.q_dist_table[:, :, :, 1] = hyperparameters.sigma_sq_init  # Initialize variances
        self.alpha = hyperparameters.alpha
        self.gamma = hyperparameters.gamma
        self.epsilon = hyperparameters.epsilon
        self.action_space = list(action_space)
        if hyperparameters.sigma_sq_init <= 0:
            raise ValueError("Initial variance (sigma_sq_init) must be positive.")
        
        if hyperparameters.obs_noise_variance <= 0:
            raise ValueError("Observation noise variance must be positive.")
        # Precision of the observation noise in the Bellman update
        self.tau_obs = 1.0 / hyperparameters.obs_noise_variance
        self.seed = hyperparameters.random_seed
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

    def thompson_sample_action(self, state: Tuple[int, int]) -> Action:
        """
        Selects an action using Thompson Sampling.
        Samples a Q-value from the distribution of each action and chooses the best.

        Args:
            state (Tuple[int, int]): The current state of the agent.

        Returns:
            int: The action selected by Thompson Sampling.
        """
        row, col = state
        mu = self.q_dist_table[row, col, :, 0]  # Means for all actions
        sigma_sq = self.q_dist_table[row, col, :, 1]  # Variances for all actions
        sampled_q_values = np.array([
            np.random.normal(mu[action], np.sqrt(sigma_sq[action]))
            for action in range(len(self.action_space))
        ])
        return Action(np.argmax(sampled_q_values))
    
    def exploit_action(self, state: Tuple[int, int]) -> Action:
        """
        Selects the action with the highest mean Q-value for the given state.

        Args:
            state (Tuple[int, int]): The current state of the agent.

        Returns:
            Action: The action with the highest mean Q-value.
        """
        row, col = state
        action_id = np.argmax(self.q_dist_table[row, col, :, 0])
        return Action(action_id)

    def choose_action(self, state: Tuple[int, int]) -> Action:
        """
        Select an action using ε-greedy strategy.

        Args:
            state (Tuple[int, int]): Current state of the agent in the maze.

        Returns:
            An Action selected either randomly or greedily.
        """
        if random.random() < self.epsilon:
            random_step = random.choice(self.action_space)
            return random_step
        return self.thompson_sample_action(state) if random.random() < 0.5 else self.exploit_action(state)

    def calculate_bellman_target(self,
                              state: Tuple[int, int],
                              action:Action,
                              reward: float,
                              next_state: Tuple[int, int],
                              done:bool) -> Tuple[float, float, float]:
        """
        Updates the Gaussian distribution for Q(state, action) based on the
        observed transition (s, a, r, s').

        Args:
            state (Tuple[int, int]): The state where the action was taken.
            action (Action): The action taken.
            reward (float): The reward received.
            next_state (Tuple[int, int]): The resulting state.
            done (bool): True if the episode terminated after this transition.
        """
        row, col = state
        next_row, next_col = next_state
        action_id = action.value

        # Current prior parameters for Q(state, action)
        mu_sa_prior = self.q_dist_table[row, col, action_id, 0]
        sigma_sq_sa_prior = self.q_dist_table[row, col, action_id, 1]

        # Handle potential numerical issues with zero variance (though unlikely with this update)
        if sigma_sq_sa_prior <= 1e-9: # Effectively infinite precision
            tau_sa_prior = 1e9
        else:
            tau_sa_prior = 1.0 / sigma_sq_sa_prior

        # Construct the target 'y' for the Bellman update
        # For the target, we sample from Q(next_state, a') to incorporate uncertainty
        if done:
            target_q_sample_max = 0.0
        else:
            sampled_next_q_values = np.array([
                np.random.normal(self.q_dist_table[next_row, next_col, next_a, 0],
                                 np.sqrt(self.q_dist_table[next_row, next_col, next_a, 1]))
                for next_a in range(len(self.action_space))
            ])
            target_q_sample_max = np.max(sampled_next_q_values)

        y = reward + self.gamma * target_q_sample_max
        return y, mu_sa_prior, tau_sa_prior

    def bayesian_update(
        self,
        state: Tuple[int, int],
        action: Action,
        y: float,
        mu_sa_prior: float,
        tau_sa_prior: float) -> None:
        """
        Bayesian update for Gaussian parameters
        """
        row, col = state
        action_id = action.value

        # New precision is prior precision + observation precision
        new_tau_sa = tau_sa_prior + self.tau_obs
        new_sigma_sq_sa = 1.0 / new_tau_sa

        # New mean is a weighted average of prior mean and observed target y
        new_mu_sa = new_sigma_sq_sa * (tau_sa_prior * mu_sa_prior + self.tau_obs * y)

        # Update the distribution parameters
        self.q_dist_table[row, col, action_id, 0] = new_mu_sa
        self.q_dist_table[row, col, action_id, 1] = tau_sa_prior * self.tau_obs * new_sigma_sq_sa
    

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
            done: True if the episode terminated after this transition.
        """
        y:float
        mu_sa_prior: float
        tau_sa_prior: float
        y, mu_sa_prior, tau_sa_prior = self.calculate_bellman_target(state, action, reward, next_state, done=done)
        self.bayesian_update(
            state,
            action,
            y,
            mu_sa_prior,
            tau_sa_prior
        )

    def decay_epsilon(self, decay_rate: float = 0.99, min_epsilon: float = 0.01) -> None:
        """
        Reduce the exploration rate over time.

        Args:
            decay_rate: Multiplicative factor to reduce epsilon.
            min_epsilon: Lower bound for epsilon.
        """
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)

    def get_dist(self, state: Tuple[int, int], action: Action) -> Tuple[float, float]:
        """
        Get the mean and variance of the Q-value distribution for a given state-action pair (useful for evaluation).

        Args:
            state (Tuple[int, int]): The current state.
            action (Action): The action taken.

        Returns:
            Tuple[float, float]: Mean and variance of the Q-value distribution.
        """
        row, col = state
        action_id = action.value
        return self.q_dist_table[row, col, action_id, 0], self.q_dist_table[row, col, action_id, 1]
