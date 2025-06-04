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
import pickle
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import asyncio


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

        if self.save_results:
            with open(os.path.join(self.storage_path, f"{self.experiment_name}_instance.pkl"), "wb") as f:
                pickle.dump(self, f)
            
        return scores


    def create_dashboard(self, experiment_result: ExperimentResult):
        maze = np.array([
            [0, 0, 1, 0],
            [1, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 1, 1, 0]
        ])
        # --- 5. Create Dash App ---
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

        # --- 6. Define Layout ---
        app.layout = dbc.Container(fluid=True, children=[
            html.H1("Experiment Results Dashboard", className="my-4 text-center"),

            dbc.Row(className="mb-4", children=[
                dbc.Col(
                    dcc.Graph(
                        id='maze-trajectory-graph',
                        figure=self.create_maze_trajectory_figure(maze, experiment_result),
                        style={'height': '500px'}
                    ),
                    lg=8 # Maze graph takes 2/3 of the width on large screens
                ),
                dbc.Col(
                    html.Div(className="card shadow-sm p-3 mb-4", children=[
                        html.H4("Experiment Metrics", className="card-title text-center mb-3"),
                        dbc.Row(children=[
                            dbc.Col(dbc.Card(dbc.CardBody([
                                html.H5("Success Rate", className="card-subtitle mb-2 text-muted"),
                                html.H3(f"{experiment_result.success_rate:.2f}", className="card-text text-center"),
                                # Optional: Add delta if you have a reference or previous value
                                # html.P(f"vs. 0.70 (Target)", className="text-muted text-center")
                            ]), className="m-1")),
                            dbc.Col(dbc.Card(dbc.CardBody([
                                html.H5("Average Reward", className="card-subtitle mb-2 text-muted"),
                                html.H3(f"{experiment_result.average_reward:.2f}", className="card-text text-center")
                            ]), className="m-1"))
                        ]),
                        dbc.Row(children=[
                            dbc.Col(dbc.Card(dbc.CardBody([
                                html.H5("Max Reward", className="card-subtitle mb-2 text-muted"),
                                html.H3(f"{experiment_result.max_reward:.2f}", className="card-text text-center")
                            ]), className="m-1")),
                            dbc.Col(dbc.Card(dbc.CardBody([
                                html.H5("Learning Speed", className="card-subtitle mb-2 text-muted"),
                                html.H3(f"{-1*experiment_result.learning_speed:.2f}", className="card-text text-center")
                            ]), className="m-1"))
                        ]),
                        dbc.Row(children=[
                            dbc.Col(dbc.Card(dbc.CardBody([
                                html.H5("Best Path Length", className="card-subtitle mb-2 text-muted"),
                                html.H3(f"{experiment_result.best_path_length}", className="card-text text-center")
                            ]), className="m-1"))
                        ], justify="center") # Center the single card if there's an odd number
                    ]),
                    lg=4 # Metrics column takes 1/3 of the width on large screens
                )
            ]),

            dbc.Row(className="mb-4", children=[
                dbc.Col(
                    dcc.Graph(
                        id='cumulative-rewards-graph',
                        figure=self.create_cumulative_rewards_figure(experiment_result),
                        style={'height': '400px'}
                    ),
                    width=12 # Cumulative rewards takes full width below
                )
            ])
        ])
        return app

    
    def create_maze_trajectory_figure(self, maze: np.ndarray, experiment_result: ExperimentResult):
        rows, cols = maze.shape
        fig = go.Figure()

        # Depicting the maze as a heatmap
        fig.add_trace(
            go.Heatmap(
                z=maze,
                colorscale=[[0, 'white'], [1, 'gray'], [0.66, 'green'], [1, 'red']], # Path, Obstacle, Start, End
                showscale=False,
                name='Maze'
            )
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
                )
            )

        # Add start and end points for clarity if they exist in the maze
        start_y, start_x = np.where(maze == 2)
        end_y, end_x = np.where(maze == 3)

        if start_y.size > 0:
            fig.add_trace(go.Scatter(
                x=[start_x[0]], y=[start_y[0]], mode='markers',
                marker=dict(symbol='star', size=15, color='green', line=dict(width=1, color='black')),
                name='Start', hoverinfo='name', showlegend=False
            ))

        if end_y.size > 0:
            fig.add_trace(go.Scatter(
                x=[end_x[0]], y=[end_y[0]], mode='markers',
                marker=dict(symbol='star', size=15, color='red', line=dict(width=1, color='black')),
                name='End', hoverinfo='name', showlegend=False
            ))

        fig.update_layout(
            title="Maze Trajectories",
            xaxis=dict(
                title="X-coordinate",
                tickmode='array',
                tickvals=np.arange(cols),
                ticktext=[str(i) for i in np.arange(cols)],
                range=[-0.5, cols - 0.5] # Extend range to show full cells
            ),
            yaxis=dict(
                title="Y-coordinate",
                tickmode='array',
                tickvals=np.arange(rows),
                ticktext=[str(i) for i in np.arange(rows)],
                autorange='reversed', # Reverse Y-axis to match typical array indexing (row 0 at top)
                range=[rows - 0.5, -0.5] # Extend range to show full cells, reversed
            ),
            plot_bgcolor='rgba(0,0,0,0)', # Transparent background
            paper_bgcolor='rgba(0,0,0,0)', # Transparent paper background
            margin=dict(l=40, r=40, t=40, b=40)
        )
        return fig

    # --- 4. Function to Create Cumulative Rewards Figure ---
    def create_cumulative_rewards_figure(self, experiment_result: ExperimentResult):
        cumulative_reward = experiment_result.cumulative_reward
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(range(len(cumulative_reward))),
                y=cumulative_reward,
                mode='lines+markers',
                name='Cumulative Reward',
                line=dict(color='purple', width=2),
                marker=dict(size=4)
            )
        )
        fig.update_layout(
            title="Cumulative Rewards Over Episodes",
            xaxis_title="Episode",
            yaxis_title="Cumulative Reward",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=40, t=40, b=40)
        )
        return fig

    