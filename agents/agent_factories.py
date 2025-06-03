from agents.q_learning import QLearningAgent
from experiment.hyperparameters import Hyperparameters
from maze.basic_maze import BasicMaze

"""
Agent Factories

This module provides factory functions for creating agent instances with specific hyperparameters.
Factories are used to keep the agent creation logic separate from the experiment logic, allowing for easy configuration and instantiation of agents.

Currently implemented:
- Q-learning agent factory

Functions:
    q_learning_factory(env: BasicMaze, hp: Hyperparameters) -> QLearningAgent
        Creates and returns a QLearningAgent configured for the given environment and hyperparameters.
"""

def q_learning_factory(env: BasicMaze, hp: Hyperparameters) -> QLearningAgent:
    return QLearningAgent(
        maze_shape = env.get_shape(),
        action_Space = env.get_action_space(),
        alpha = hp.alpha,
        gamma = hp.gamma,
        epsilon = hp.epsilon
    )
