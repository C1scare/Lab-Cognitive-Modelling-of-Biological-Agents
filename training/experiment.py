import os
import numpy as np
from maze.basic_maze import BasicMaze, GameStatus
from agents.q_learning import QLearningAgent
from agents.bayesian_q_learning import BayesianQLearningAgent
from training.train_script import train_agent
import matplotlib.pyplot as plt
from enums.agent_type import AgentType
from agents.noisy_agent import NoisyAgent
from agents.curious_agent import CuriousAgent
from enums.noise_mode import NoiseMode
from training.hyperparameter import Hyperparameter
from training.train_script import ExperimentResult
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
        maze_array = np.array([
            [0, 0, 1, 0],
            [1, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 1, 1, 0]
        ])

        env = BasicMaze(maze=maze_array, start_cell=(0, 0), goal_cell=(3, 3))

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

        scores:ExperimentResult = train_agent(env, agent, episodes=self.hyperparameters.episodes)

        return scores


    def create_dashboard(self, experiment_result: ExperimentResult):
        maze = np.array([
            [0, 0, 1, 0],
            [1, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 1, 1, 0]
        ])
        rows, cols = maze.shape

        fig = make_subplots(
            rows=2,
            cols=2,
            column_widths=[0.6, 0.4],
            row_heights=[0.6, 0.4],
            specs=[
                [{"type": "xy", "rowspan": 1, "colspan": 1}, {"type": "indicator", "rowspan": 1, "colspan": 1}],
                [{"type": "xy", "rowspan": 1, "colspan": 1}, {"type": "indicator", "rowspan": 1, "colspan": 1}]
            ],
            subplot_titles=("Maze Trajectories", "Experiment Metrics", "Cumulative Rewards", "Additional Metrics")
        )

        # --- 1. Maze and Trajectories ---
        # Depicting the maze
        # Use an image trace for the maze for better visual representation
        maze_img = np.zeros((rows, cols, 3), dtype=np.uint8)
        maze_img[maze == 0] = [255, 255, 255]  # Path (white)
        maze_img[maze == 1] = [100, 100, 100]  # Obstacle (gray)
        maze_img[maze == 2] = [0, 255, 0]    # Start (green)
        maze_img[maze == 3] = [255, 0, 0]    # End (red)

        fig.add_trace(
            go.Heatmap(z=maze, colorscale=[[0, 'white'], [1, 'gray'], [0.66, 'green'], [1, 'red']], showscale=False),
            row=1, col=1
        )

        # Plotting trajectories with low opacity and hover info
        for episode, trajectory in experiment_result.trajectory_history.items():
            x_coords = [point[1] for point in trajectory]
            y_coords = [point[0] for point in trajectory]
            hover_text = [f"Episode: {episode}<br>Step: {i}<br>Position: ({y},{x})" for i, (y, x) in enumerate(trajectory)]

            fig.add_trace(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='lines+markers',
                    name=f'Episode {episode}',
                    line=dict(width=1, color='blue'),
                    marker=dict(size=4, opacity=0.3),
                    opacity=0.1,  # Low opacity for distribution
                    hoverinfo='text',
                    hovertext=hover_text,
                    showlegend=False
                ),
                row=1, col=1
            )

        # Add start and end points for clarity if they exist in the maze
        start_y, start_x = np.where(maze == 2)
        end_y, end_x = np.where(maze == 3)

        if start_y.size > 0:
            fig.add_trace(go.Scatter(
                x=[start_x[0]], y=[start_y[0]], mode='markers',
                marker=dict(symbol='star', size=15, color='green'),
                name='Start', hoverinfo='name', showlegend=False
            ), row=1, col=1)

        if end_y.size > 0:
            fig.add_trace(go.Scatter(
                x=[end_x[0]], y=[end_x[0]], mode='markers',
                marker=dict(symbol='star', size=15, color='red'),
                name='End', hoverinfo='name', showlegend=False
            ), row=1, col=1)


        fig.update_xaxes(title_text="X-coordinate", row=1, col=1,
                        tickmode='array', tickvals=np.arange(cols), ticktext=[str(i) for i in np.arange(cols)])
        fig.update_yaxes(title_text="Y-coordinate", row=1, col=1,
                        tickmode='array', tickvals=np.arange(rows), ticktext=[str(i) for i in np.arange(rows)],
                        autorange='reversed') # Reverse Y-axis to match typical array indexing (row 0 at top)


        # --- 2. Cumulative Rewards Graph ---
        fig.add_trace(
            go.Scatter(
                x=list(range(len(experiment_result.cumulative_reward))),
                y=experiment_result.cumulative_reward,
                mode='lines+markers',
                name='Cumulative Reward Over Episodes',
                line=dict(color='purple', width=2),
                marker=dict(size=4)
            ),
            row=2, col=1
        )
        fig.update_xaxes(title_text="Episode", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative Reward", row=2, col=1)

        # --- 3. Other Metric Information (Indicators) ---
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=experiment_result.success_rate,
                number={'valueformat': ".2f"},
                delta={'reference': 0.7, 'relative': False, 'valueformat': ".2f", 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
                title={"text": "Success Rate"},
            ),
            row=1, col=2
        )

        # To add multiple indicators in the same subplot without them overlapping too much,
        # you often need to define their exact positions within the subplot's domain.
        # Alternatively, you can create more subplots for indicators or use annotations.
        # For simplicity, let's stack them vertically using a slightly different approach or just accept default stacking.

        # Instead of adding all indicators to row=2, col=2 directly,
        # let's manually position them if we want to avoid them stacking directly on top of each other.
        # However, make_subplots automatically positions them within the given cell.
        # If you want them separated, you need to add more cells or adjust domain.

        # Here, I'll put additional metrics as separate indicators but still in the same subplot area,
        # relying on Plotly's default stacking. For better side-by-side display,
        # you'd need more subplot cells or a more complex layout.
        # The existing code adds them to (2,2) and (1,2). Let's refine how they are displayed.

        # Let's add the remaining indicators to the existing indicator subplot area (1,2)
        # or consider making a new subplot area for them to separate concerns if needed.
        # For now, let's add them to the same column (col=2) and let Plotly stack them.

        fig.add_trace(
            go.Indicator(
                mode="number",
                value=experiment_result.average_reward,
                number={'valueformat': ".2f"},
                title={"text": "Average Reward"},
            ),
            row=1, col=2 # Keep adding to this same indicator subplot area
        )

        fig.add_trace(
            go.Indicator(
                mode="number",
                value=experiment_result.max_reward,
                number={'valueformat': ".2f"},
                title={"text": "Max Reward"},
            ),
            row=1, col=2 # Keep adding to this same indicator subplot area
        )

        fig.add_trace(
            go.Indicator(
                mode="number",
                value=experiment_result.learning_speed,
                number={'valueformat': ".2f"},
                title={"text": "Learning Speed"},
            ),
            row=1, col=2 # Keep adding to this same indicator subplot area
        )

        fig.add_trace(
            go.Indicator(
                mode="number",
                value=experiment_result.best_path_length,
                number={'valueformat': "d"},
                title={"text": "Best Path Length"},
            ),
            row=1, col=2 # Keep adding to this same indicator subplot area
        )


        fig.update_layout(
            height=800,
            width=1200,
            title_text="Experiment Results Dashboard",
            template="plotly_white",
        )

        # Update subplot title font size
        for annotation in fig.layout.annotations:
            if 'subplot' in annotation.text: # Check if it's a subplot title
                annotation.font.size = 16 # Set desired font size

        return fig