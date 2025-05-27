from agents.bayesian_q_learning import BayesianQLearningAgent
import numpy as np
from typing import Tuple, Sequence
from maze.basic_maze import Action
from agents.noise_mode import NoiseMode


class NoisyAgent(BayesianQLearningAgent):
    """
    A Noisy Q-learning agent that incorporates noise in the Q-value updates.
    
    Inherits from BayesianQLearningAgent to utilize its structure and methods,
    while adding noise to the Q-value updates.
    """
    def __init__(
        self, 
        maze_shape: Tuple[int, int], 
        action_space: Sequence[Action], 
        alpha: float = 0.1, 
        gamma: float = 0.99, 
        epsilon: float = 0.2,
        mu_init: float = 0.0,
        sigma_sq_init: float = 2.0,
        obs_noise_variance: float = 0.1,
        k_pn: float = 0.1,
        sigma_nn: float = 1,
        noise_mode: NoiseMode = NoiseMode.BOTH
    ) -> None:
        """
        Initialize the Noisy Q-learning agent with an empty Q-table and noise parameters.
        
        Args:
            maze_shape: Dimensions of the maze grid.
            action_space: List of possible actions (e.g., UP, DOWN, etc.).
            alpha: Learning rate for Q-value updates.
            gamma: Discount factor for future rewards.
            epsilon: Initial exploration rate.
            mu_init: Initial mean for the Gaussian Q-value distributions.
            sigma_sq_init: Initial variance for the Gaussian Q-value distributions.
            obs_noise_variance: Assumed variance of the Bellman target 'y'.
            k_pn: Parameter for the magnitude of perceptual noise
            sigma_nn: variance of neural noise proportional to the magnitude of neural noise
            noise_mode: Mode of noise to apply (e.g., both perceptual and neural).
        """
        super().__init__(
            maze_shape=maze_shape,
            action_space=action_space,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            mu_init=mu_init,
            sigma_sq_init=sigma_sq_init,
            obs_noise_variance=obs_noise_variance
        )
        self.k_pn = k_pn
        self.sigma_nn = sigma_nn 
        self.noise_mode = noise_mode
        if noise_mode == NoiseMode.BOTH or noise_mode == NoiseMode.NEURAL and sigma_nn <= 0:
            raise ValueError("Neural noise variance (sigma_nn) must be greater than 0 when neural noise is enabled.")

    def update_q_distribution(self,
                              state: Tuple[int, int],
                              action:Action,
                              reward: float,
                              next_state: Tuple[int, int],
                              done:bool) -> None:
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
        if self.noise_mode == NoiseMode.NEURAL or self.noise_mode == NoiseMode.BOTH:
            # Apply neural noise to the target value
            y = self.apply_neural_noise(y)

        # Bayesian update for Gaussian parameters
        # New precision is prior precision + observation precision
        new_tau_sa = tau_sa_prior + self.tau_obs
        new_sigma_sq_sa = 1.0 / new_tau_sa

        # New mean is a weighted average of prior mean and observed target y
        new_mu_sa = new_sigma_sq_sa * (tau_sa_prior * mu_sa_prior + self.tau_obs * y)

        # Update the distribution parameters
        self.q_dist_table[row, col, action_id, 0] = new_mu_sa
        self.q_dist_table[row, col, action_id, 1] = new_sigma_sq_sa

    def apply_perceptual_noise(self, reward: float) -> float:
        """
        Apply perceptual noise to the reward based on the k_pn parameter.
        Args:
            reward: The original reward value.
        Returns:
            The reward value after applying perceptual noise.
        """
        mu_pn = reward
        sigma_pn = self.k_pn * abs(mu_pn)
        return np.random.normal(mu_pn, sigma_pn)

    def apply_neural_noise(self, q_value: float) -> float:
        """
        Apply neural noise to the Q-value based on the sigma_nn parameter.
        Args:
            q_value: The original Q-value.
        Returns:
            The Q-value after applying neural noise.
        """
        return np.random.normal(q_value, self.sigma_nn)
        
    def learn(
        self, 
        state: Tuple[int, int],
        action: Action, 
        reward: float, 
        next_state: Tuple[int, int],
        done: bool = False
    ) -> None:
        """
        Update the Q-value for a given state-action pair with added noise.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received after taking the action.
            next_state: Next state after taking the action.
            done: Whether the episode has ended.
        """
        if self.noise_mode == NoiseMode.PERCEPTUAL or self.noise_mode == NoiseMode.BOTH:
            # Apply perceptual noise to the reward
            reward = self.apply_perceptual_noise(reward)
        super().learn(state, action, reward, next_state, done)