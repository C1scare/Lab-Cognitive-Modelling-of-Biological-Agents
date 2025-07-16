from agents.bayesian_q_learning import BayesianQLearningAgent
import numpy as np
from typing import Tuple, Sequence
from maze.basic_maze import Action
from training.hyperparameter import Hyperparameter
import random


class CuriousAgent(BayesianQLearningAgent):
    """
    A Curious Q-learning agent that incorporates curiosity-driven exploration.
    
    Inherits from BayesianQLearningAgent to utilize its structure and methods,
    while adding curiosity-driven exploration strategies.
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
            curiosity_init=0.5,
            alpha_C=0.1,
            surprise_weight=1.0,
            novelty_weight=1.0,
            usefulness_weight=1.0,
            uncertainty_weight=1.0,
            alpha_tau=0.1
        )
    ) -> None:
        """
        Initialize the Curious Q-learning agent with an empty Q-table and curiosity parameters.
        
        Args:
            maze_shape: Dimensions of the maze grid.
            action_space: List of possible actions (e.g., UP, DOWN, etc.).
            alpha: Learning rate for Q-value updates.
            gamma: Discount factor for future rewards.
            epsilon: Initial exploration rate.
            mu_init: Initial mean for the Gaussian Q-value distributions.
            sigma_sq_init: Initial variance for the Gaussian Q-value distributions.
            obs_noise_variance: Assumed variance of the Bellman target 'y'.
        """
        super().__init__(
            maze_shape=maze_shape,
            action_space=action_space,
            hyperparameters=hyperparameters
        )
        if (hyperparameters.curiosity_init is None or
            hyperparameters.alpha_C is None or
            hyperparameters.surprise_weight is None or
            hyperparameters.novelty_weight is None or
            hyperparameters.usefulness_weight is None or
            hyperparameters.uncertainty_weight is None or
            hyperparameters.alpha_tau is None):
            raise ValueError("Curiosity initialization value must be provided.")
        
        self.curiosity = np.ones((*maze_shape, len(action_space))) * hyperparameters.curiosity_init
        self.visited = np.zeros(maze_shape)
        self.alpha_C = hyperparameters.alpha_C
        self.surprise_weight = hyperparameters.surprise_weight
        self.novelty_weight = hyperparameters.novelty_weight
        self.usefulness_weight = hyperparameters.usefulness_weight
        self.uncertainty_weight = hyperparameters.uncertainty_weight
        self.alpha_tau = hyperparameters.alpha_tau
        self.seed = hyperparameters.random_seed
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

    def calculate_surprise(self,
                           state:Tuple[int, int],
                           action:Action,
                           y:float) -> float:
        """
        Calculate the surprise value for a given state-action pair.
        """
        mu = self.q_dist_table[state[0], state[1], action.value, 0]
        tau = self.q_dist_table[state[0], state[1], action.value, 1]
        return abs(y - mu) / (tau + 1)

    def calculate_novelty(self, next_state:Tuple[int, int]) -> float:
        """
        Calculate the novelty of a state based on how many times it has been visited.
        The novelty is inversely proportional to the number of visits to the state.
        """
        return 1 / np.sqrt(self.visited[next_state] + 1)

    def calculate_usefulness(self, state:Tuple[int, int], action:Action, y:float) -> float:
        """
        Calculate the usefulness of a state-action pair based on the Q-value distribution.
        This is typically represented by the mean of the Q-value distribution.
        """
        return max(0, y - self.q_dist_table[state[0], state[1], action.value, 0])

    def calculate_uncertainty(self, state:Tuple[int, int], action:Action) -> float:
        """
        Calculate the uncertainty of the Q-value distribution for a given state-action pair.
        This is typically represented by the standard deviation of the Q-value distribution.
        """
        return min(1, np.sqrt(self.q_dist_table[state[0], state[1], action.value, 1]))

    def calculate_curiosity(self,
                            state:Tuple[int, int],
                            action:Action,
                            next_state:Tuple[int, int],
                            y:float
                            ) -> float:
        """
        Calculate the curiosity-driven exploration value based on surprise, novelty, usefulness, and uncertainty.
        
        This method should be implemented to define how curiosity is quantified in the context of the agent's learning.
        """
        # Calculate individual components of curiosity
        surprise = self.calculate_surprise(state, action, y)
        novelty = self.calculate_novelty(next_state)
        usefulness = self.calculate_usefulness(state, action, y)
        uncertainty = self.calculate_uncertainty(state, action)
            
        # Combine these components with their respective weights
        component_contributions = (self.surprise_weight * surprise +
                                    self.novelty_weight * novelty +
                                    self.usefulness_weight * usefulness +
                                    self.uncertainty_weight * uncertainty)

        curiosity_value = (1-self.alpha_C) * self.curiosity[*state, action.value] + \
                          self.alpha_C * component_contributions
        self.curiosity[*state, action.value] = curiosity_value

        return curiosity_value
    
    def choose_action(self, state: Tuple[int, int]) -> Action:
        """
        Select an action based on the current state and the agent's policy.
        Args:
            state (Tuple[int, int]): The current state of the agent in the maze.
        Returns:
            Action: The action selected by the agent.
        """
        '''
        epsilon = min(1.0, self.epsilon + (1-self.alpha_tau) * np.max(self.curiosity[state,:]))
        if random.random() < epsilon:
            return random.choice(self.action_space)
        return self.thompson_sample_action(state)
        '''
        if random.random() < self.epsilon:
            # Select a random action with probability epsilon
            return random.choice(self.action_space)
        return self.curious_thompson_sample_action(state)
        #'''
    
    def curious_thompson_sample_action(self, state: Tuple[int, int]) -> Action:
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
        sigma_sq = self.q_dist_table[row, col, :, 1] * (self.alpha_tau * np.abs(self.curiosity[row, col, :])+1)  # Variances for all actions
        sampled_q_values = np.array([
            np.random.normal(mu[action], np.sqrt(sigma_sq[action]))
            for action in range(len(self.action_space))
        ])
        return Action(np.argmax(sampled_q_values))
    

    def learn(self, state: Tuple[int, int], action: Action, reward: float, next_state: Tuple[int, int], done: bool) -> None:
        """
        Update the Q-value distribution based on the observed transition (s, a, r, s').
        
        Args:
            state (Tuple[int, int]): The state where the action was taken.
            action (Action): The action taken.
            reward (float): The reward received after taking the action.
            next_state (Tuple[int, int]): The next state after taking the action.
            done (bool): Whether the episode has ended.
        """
        y, mu_sa_prior, tau_sa_prior = super().calculate_bellman_target(state, action, reward, next_state, done)
        _ = self.calculate_curiosity(state, action, next_state, y)
        self.bayesian_update(state, action, y, mu_sa_prior, tau_sa_prior)
        self.visited[state] += 1
