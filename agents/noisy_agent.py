from agents.bayesian_q_learning import BayesianQLearningAgent
import numpy as np
from typing import Tuple, Sequence
from maze.basic_maze import Action
from enums.noise_mode import NoiseMode
from training.hyperparameter import Hyperparameter


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
        hyperparameters: Hyperparameter = Hyperparameter(
            gamma=0.99, 
            epsilon=0.2,
            mu_init=0.0,
            sigma_sq_init=2.0,
            obs_noise_variance=0.1,
            k_pn=0.1,
            sigma_nn=1,
            noise_mode=NoiseMode.BOTH
        )
    ) -> None:
        """
        Initialize the Noisy Q-learning agent with an empty Q-table and noise parameters.
        
        Args:
            maze_shape: Dimensions of the maze grid.
            action_space: List of possible actions (e.g., UP, DOWN, etc.).
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
            hyperparameters=hyperparameters
        )
        if (hyperparameters.k_pn is None or
            hyperparameters.sigma_nn is None or
            hyperparameters.noise_mode is None):
            raise ValueError("Hyperparameters must be provided with valid values.")

        self.k_pn = hyperparameters.k_pn
        self.sigma_nn = hyperparameters.sigma_nn 
        self.noise_mode = hyperparameters.noise_mode
        if hyperparameters.noise_mode == NoiseMode.BOTH or hyperparameters.noise_mode == NoiseMode.NEURAL and self.sigma_nn <= 0:
            raise ValueError("Neural noise variance (sigma_nn) must be greater than 0 when neural noise is enabled.")

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

        y:float
        mu_sa_prior: float
        tau_sa_prior: float
        y, mu_sa_prior, tau_sa_prior = self.calculate_bellman_target(state, action, reward, next_state, done=done)

        if self.noise_mode == NoiseMode.NEURAL or self.noise_mode == NoiseMode.BOTH:
            # Apply neural noise to the target value
            y = self.apply_neural_noise(y)

        self.bayesian_update(
            state,
            action,
            y,
            mu_sa_prior,
            tau_sa_prior
        )