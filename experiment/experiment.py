import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Callable, Optional
from maze.basic_maze import BasicMaze, GameStatus
from maze.maze_definitions import mazes
from agents.base_agent import BaseAgent
from experiment.hyperparameters import Hyperparameters
from training.train_script import train_agent, plot_rewards
import matplotlib.pyplot as plt


class Experiment:
    """
    Manages a single reinforcement learning experiment run.
    
    This class handles environment setup, agent instantiation (from factory or pre-trained file),
    training, evaluation, and result visualization. It automatically organizes output data
    (such as trained agents and reward plots) in a structured folder hierarchy under `results/`.

    Directory structure:
        results/
        └── AgentClassName/
            └── <base_name><timestamp>/
                ├── config.json
                ├── agent_trained.pkl
                └── (plots, logs, etc.)
    """

    def __init__(
        self,
        hyperparameters: Hyperparameters,
        save_dir: Optional[str] = None,
        agent_factory: Optional[Callable[[BasicMaze, Hyperparameters], BaseAgent]] = None,
        agent_path: Optional[str] = None
    ):
        """
        Initialize the Experiment.

        Args:
            hyperparameters (Hyperparameters): Configuration for training and environment.
            save_dir (Optional[str]): Optional base name for the experiment folder. 
                                      A timestamp is appended automatically.
                                      If omitted, defaults to "<maze_name>_start<start_cell_idx>_".
            agent_factory (Optional[Callable]): Function that returns a new untrained agent when called with
                                                the environment and hyperparameters.
            agent_path (Optional[str]): Optional path to a pre-trained agent (overrides agent_factory if provided).

        Raises:
            ValueError:If neither agent_factory nor agent_path is provided.
        """
        self.hyperparameters = hyperparameters
        self.maze_info = mazes[hyperparameters.maze_name]
        self.agent_factory: Optional[Callable[[BasicMaze, Hyperparameters], BaseAgent]] = agent_factory
        self.agent_path = agent_path
        self.agent: BaseAgent
        self.rewards: List[float] = []

        # Temporarily load agent (or determine agent class later)
        self._agent_class_name = None
        if agent_path:
            with open(agent_path, "rb") as f:
                agent = pickle.load(f)
            self._agent_class_name = agent.__class__.__name__
        elif agent_factory:
            # Use the return type of a dummy agent for class name
            dummy_env = BasicMaze(
                maze=self.maze_info["maze"],
                start_cell=self.maze_info["start_cells"][0],
                goal_cell=self.maze_info["goal_cell"],
                max_steps=1
            )
            dummy_agent = agent_factory(dummy_env, hyperparameters)
            self._agent_class_name = dummy_agent.__class__.__name__
        else:
            raise ValueError("Either agent_path or agent_factory must be provided.")

        # Format save_dir
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_name = save_dir or ""
        self.save_dir = Path("results") / self._agent_class_name / f"{base_name}{timestamp}"

    def run(
            self,
            train: bool = True, 
            show_plot: bool = True, 
            show_animation: bool = True
        ) -> None:
        """
        Run the experiment by training the agent and/or visualizing performance.

        Args:
            train (bool): Whether to train the agent. If False and a pre-trained agent is provided,
                          only evaluation/visualization is done.
            show_plot (bool): Whether to plot the episode reward curve after training.
            show_animation (bool): Whether to run an animated test episode after training/evaluation.

        Raises:
            ValueError: If neither a pre-trained agent nor a factory function is available to create one.
        """
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.hyperparameters.to_json(str(self.save_dir / "config.json"))

        self.env = BasicMaze(
            maze=self.maze_info["maze"],
            start_cell=self.maze_info["start_cells"][self.hyperparameters.start_cell_idx],
            goal_cell=self.maze_info["goal_cell"],
            max_steps=self.hyperparameters.max_steps
        )

        if self.agent_path:
            with open(self.agent_path, "rb") as f:
                self.agent = pickle.load(f)
            print(f"Loaded agent from {self.agent_path}")
        elif self.agent_factory:
            self.agent = self.agent_factory(self.env, self.hyperparameters)
        else:
            raise ValueError("Either agent_path or agent_factory must be provided.")

        if train:
            self.rewards = train_agent(
                self.env,
                self.agent,
                episodes=self.hyperparameters.episodes,
                decay_epsilon=self.hyperparameters.decay_epsilon
            )
            self.agent.save_agent(str(self.save_dir / "agent_trained.pkl"))

        if show_plot:
            if self.rewards:
                plot_rewards(self.rewards)
        if show_animation:
            self.state = self.env.reset(self.env.start_cell)
            plt.ion()
            self.fig, self.ax = plt.subplots()
            self.env.render()
            plt.pause(0.3)

            while True:
                action = self.agent.choose_action(self.state)
                self.state, _, self.status = self.env.step(action)
                self.env.render()
                plt.pause(0.1)
                if self.status != GameStatus.IN_PROGRESS:
                    print("Test run ended with status:", self.status)
                    break

            plt.ioff()
            plt.show()
