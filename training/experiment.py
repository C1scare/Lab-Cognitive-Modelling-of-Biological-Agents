import os
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
import pickle
from maze.maze_scheduler import MazeScheduler


class Experiment:
    """
    Manages the setup, execution, and result handling for a maze-based RL experiment.

    This class initializes the environment, agent, and hyperparameters, runs training,
    and provides methods for saving results and launching a dashboard.

    Attributes:
        experiment_name: Name of the experiment.
        storage_path: Directory to store experiment results.
        agent_type: Enum specifying which agent to use.
        hyperparameters: Hyperparameter object for agent/environment configuration.
        maze_scheduler: MazeScheduler instance for managing mazes and start positions.
        save_results: Whether to save the experiment instance after training.
    """

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
                 maze_scheduler: MazeScheduler = MazeScheduler(first=26, last=27,trials_maze=200),
                 save_results: bool = False
                 ):
        """
        Initialize the experiment with configuration and storage options.

        Args:
            experiment_name: Name of the experiment.
            storage_path: Path to store experiment results.
            agent_type: Which agent to use (Q_LEARNING, BAYESIAN_Q_LEARNING, etc.).
            hyperparameters: Hyperparameter object for agent/environment.
            maze_scheduler: MazeScheduler instance for maze management.
            save_results: Whether to save the experiment instance after training.
        """
        self.experiment_name = experiment_name
        self.storage_path = storage_path
        # Ensure the storage path exists
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)
        self.agent_type = agent_type
        self.hyperparameters = hyperparameters
        self.maze_scheduler = maze_scheduler
        self.save_results = save_results

    def run_experiment(self) -> ExperimentResult:
        """
        Run the experiment using the configured agent and environment.

        Returns:
            ExperimentResult: Object containing training metrics and histories.

        Raises:
            ValueError: If an unsupported agent type is specified.
            ValueError: If the number of episodes is not specified in hyperparameters.
        """
        env:BasicMaze = self.maze_scheduler.maze

        if self.agent_type == AgentType.Q_LEARNING:
            print("Using Q-learning agent")
            agent = QLearningAgent(
                maze_shape=env.get_shape(),
                action_Space=list(env.actions.keys()),
                hyperparameters=self.hyperparameters
            )
        
        elif self.agent_type == AgentType.BAYESIAN_Q_LEARNING:
            print("Using Bayesian Q-learning agent")
            agent = BayesianQLearningAgent(
                maze_shape=env.get_shape(),
                action_space=list(env.actions.keys()),
                hyperparameters=self.hyperparameters
            )
        elif self.agent_type == AgentType.NOISY_AGENT:
            noise_mode = NoiseMode.PERCEPTUAL  # Change to NoiseMode.NEURAL or NoiseMode.BOTH as needed
            print(f"Using Noisy agent with {noise_mode.value}")
            agent = NoisyAgent(
                maze_shape=env.get_shape(),
                action_space=list(env.actions.keys()),
                hyperparameters=self.hyperparameters,
            )
        elif self.agent_type == AgentType.CURIOUS_AGENT:
            print("Using Curious agent")
            agent = CuriousAgent(
                maze_shape=env.get_shape(),
                action_space=list(env.actions.keys()),
                hyperparameters=self.hyperparameters
            )
        elif self.agent_type == AgentType.SR_DYNA_AGENT:
            from agents.sr_dyna import SRDynaAgent
            print("Using SR-Dyna agent")
            agent = SRDynaAgent(
                maze_shape=env.get_shape(),
                action_space=list(env.actions.keys()),
                hyperparameters=self.hyperparameters
            )

        else:
            raise ValueError("Unsupported agent type specified.")

        if self.hyperparameters.episodes is None:
            raise ValueError("Number of episodes must be specified in hyperparameters.")
        episodes = self.hyperparameters.episodes
        self.scores:ExperimentResult = train_agent(self.maze_scheduler, agent, episodes=episodes)

        if self.save_results:
            with open(os.path.join(self.storage_path, f"{self.experiment_name}_instance.pkl"), "wb") as f:
                pickle.dump(self, f)
            
        return self.scores


    def create_dashboard(self, experiment_result: ExperimentResult):
        """
        Create an interactive dashboard for the experiment results.

        Args:
            experiment_result: The ExperimentResult object to visualize.

        Returns:
            A Dash app instance for interactive exploration.
        """
        from maze.maze_visualizer import MazeVisualizer
        return MazeVisualizer().create_dashboard(experiment_result, self)
