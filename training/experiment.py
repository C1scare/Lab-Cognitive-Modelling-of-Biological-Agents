import os
import numpy as np
from maze.basic_maze import BasicMaze
from agents.q_learning import QLearningAgent
from agents.bayesian_q_learning import BayesianQLearningAgent
from training.train_script import train_agent
from enums.agent_type import AgentType
from agents.noisy_agent import NoisyAgent
from agents.curious_agent import CuriousAgent
from enums.noise_mode import NoiseMode
from training.hyperparameter import Hyperparameter
from training.train_script import ExperimentResult
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import pickle
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from maze.maze_scheduler import MazeScheduler


class Experiment:
    def __init__(self,
                 experiment_name: str = "Maze Experiment",
                 storage_path: str = "experiments/",
                 agent_type: AgentType = AgentType.BAYESIAN_Q_LEARNING,
                 hyperparameters: Hyperparameter = Hyperparameter(
                     alpha=0.1,
                     gamma=0.99,
                     epsilon=0.2,
                     mu_init=0.0,
                     sigma_sq_init=2.0,
                     obs_noise_variance=0.1
                 ),
                 MazeScheduler: MazeScheduler = MazeScheduler(trials=10),
                 save_results: bool = False
                 ):
        """
        Initialize the experiment with a name and storage path.
        Args:
            experiment_name: Name of the experiment.
            storage_path: Path to store experiment results.
        """
        self.experiment_name = experiment_name
        self.storage_path = storage_path
        # Ensure the storage path exists
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)
        self.agent_type = agent_type
        self.hyperparameters = hyperparameters
        self.maze_scheduler = MazeScheduler
        self.save_results = save_results

    def run_experiment(self) -> ExperimentResult:
        """
        Run an experiment with the specified maze and agent type.

        Args:
            agent_type: The type of agent to use for the experiment.
            episodes: Number of training episodes to run.
        
        Returns:
            Dict of score metrics
        """
        env:BasicMaze = self.maze_scheduler.maze

        if self.agent_type == AgentType.Q_LEARNING:
            print("Using Q-learning agent")
            agent = QLearningAgent(
                maze_shape=env.get_shape(),
                action_Space=env.actions,
                hyperparameters=self.hyperparameters
            )
        
        elif self.agent_type == AgentType.BAYESIAN_Q_LEARNING:
            print("Using Bayesian Q-learning agent")
            agent = BayesianQLearningAgent(
                maze_shape=env.get_shape(),
                action_space=env.actions,
                hyperparameters=self.hyperparameters
            )
        elif self.agent_type == AgentType.NOISY_AGENT:
            noise_mode = NoiseMode.PERCEPTUAL  # Change to NoiseMode.NEURAL or NoiseMode.BOTH as needed
            print(f"Using Noisy agent with {noise_mode.value}")
            agent = NoisyAgent(
                maze_shape=env.get_shape(),
                action_space=env.actions,
                hyperparameters=self.hyperparameters,
            )
        elif self.agent_type == AgentType.CURIOUS_AGENT:
            print("Using Curious agent")
            agent = CuriousAgent(
                maze_shape=env.get_shape(),
                action_space=env.actions,
                hyperparameters=self.hyperparameters
            )
        else:
            raise ValueError("Unsupported agent type specified.")

        scores:ExperimentResult = train_agent(self.maze_scheduler, agent, episodes=self.hyperparameters.episodes)

        if self.save_results:
            with open(os.path.join(self.storage_path, f"{self.experiment_name}_instance.pkl"), "wb") as f:
                pickle.dump(self, f)
            
        return scores


    def create_dashboard(self, experiment_result: ExperimentResult):
        from maze.maze_visualizer import MazeVisualizer
        return MazeVisualizer().create_dashboard(experiment_result, self)
    