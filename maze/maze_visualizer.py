import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter, defaultdict
import hashlib

# Dash imports
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from training.train_script import ExperimentResult
from training.experiment import Experiment


class MazeVisualizer:
    """
    A class to handle the visualization of a maze and agent trajectories,
    including episode-specific views and averaged views across maze configurations.
    """

    def create_maze_trajectory_figure(self, maze: np.ndarray, experiment_result: 'ExperimentResult') -> go.Figure:
        """
        [DEPRECATED/MODIFIED] This function is now superseded by the new averaged views.
        Kept for reference but not directly used in the new dashboard structure.
        The core logic for plotting transitions is reused in the new functions.
        """
        rows, cols = maze.shape
        fig = go.Figure()

        fig.add_trace(
            go.Heatmap(
                z=maze,
                colorscale=[[0.0, 'white'], [0.333, 'gray'], [0.666, 'green'], [1.0, 'red']],
                zmin=0, zmax=3,
                showscale=False,
                name='Maze Grid'
            )
        )

        transition_counts = Counter()
        for episode, trajectory_data in experiment_result.trajectory_history.items():
            for current_state, next_state in trajectory_data:
                transition_counts[(current_state, next_state)] += 1

        max_count = max(transition_counts.values()) if transition_counts else 1
        colorscale_plasma = px.colors.sequential.Plasma

        for (current_y, current_x), (next_y, next_x) in transition_counts:
            count = transition_counts[((current_y, current_x), (next_y, next_x))]
            color_idx = min(int((count / max_count) * (len(colorscale_plasma) - 1)), len(colorscale_plasma) - 1)
            edge_color = colorscale_plasma[color_idx]
            line_width = 1 + (count / max_count) * 5

            fig.add_trace(
                go.Scatter(
                    x=[current_x, next_x],
                    y=[current_y, next_y],
                    mode='lines',
                    line=dict(color=edge_color, width=line_width),
                    hoverinfo='text',
                    hovertext=f"Transition: ({current_y},{current_x}) -> ({next_y},{next_x})<br>Frequency: {count}",
                    showlegend=False,
                    name=f'Transition from ({current_y},{current_x}) to ({next_y},{next_x})'
                )
            )

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
            title="Maze Trajectories with Transition Frequencies",
            xaxis=dict(
                title="X-coordinate", tickmode='array', tickvals=np.arange(cols),
                ticktext=[str(i) for i in np.arange(cols)], range=[-0.5, cols - 0.5],
                showgrid=True, zeroline=False
            ),
            yaxis=dict(
                title="Y-coordinate", tickmode='array', tickvals=np.arange(rows),
                ticktext=[str(i) for i in np.arange(rows)], autorange='reversed',
                range=[rows - 0.5, -0.5], showgrid=True, zeroline=False,
                scaleanchor="x", scaleratio=1
            ),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=40, t=40, b=40)
        )
        return fig

    def create_episode_view_with_slider(self, experiment_result: ExperimentResult) -> go.Figure:
        """
        Creates a Plotly figure with a slider to select and display individual episode trajectories.
        Each frame shows the maze, start point, and the trajectory for the selected episode.
        """
        fig = go.Figure()
        frames = []
        steps = []

        # Get all unique episode numbers and sort them for consistent slider order
        sorted_episodes = sorted(experiment_result.trajectory_history.keys())

        if not sorted_episodes:
            fig.update_layout(title="No episode data available to display.",
                              xaxis_visible=False, yaxis_visible=False)
            return fig

        # Determine max rows/cols across all mazes to ensure consistent plot sizing across frames
        max_rows = 0
        max_cols = 0
        for episode in sorted_episodes:
            if episode in experiment_result.maze_history:
                maze_obj, _ = experiment_result.maze_history[episode]
                rows, cols = maze_obj.maze.shape
                max_rows = max(max_rows, rows)
                max_cols = max(max_cols, cols)
            else:
                print(f"Warning: Maze history missing for episode {episode}. Skipping.")
                continue

        # Create frames for each episode
        for i, episode in enumerate(sorted_episodes):
            if episode not in experiment_result.maze_history or episode not in experiment_result.trajectory_history:
                continue

            maze_obj, start_location = experiment_result.maze_history[episode]
            maze = maze_obj.maze
            trajectory = experiment_result.trajectory_history[episode]

            current_frame_data = []

            current_frame_data.append(
                go.Heatmap(
                    z=maze,
                    colorscale=[
                        [0.0, 'white'], [0.333, 'gray'], [0.666, 'green'], [1.0, 'red']
                    ],
                    zmin=0, zmax=3,
                    showscale=False,
                    name='Maze Grid'
                )
            )

            line_x = []
            line_y = []
            hover_lines = []
            for j, (current_state, next_state) in enumerate(trajectory):
                line_x.extend([current_state[1], next_state[1], None])
                line_y.extend([current_state[0], next_state[0], None])
                hover_lines.append(f"Episode: {episode}<br>Step: {j}<br>Transition: ({current_state[0]},{current_state[1]}) -> ({next_state[0]},{next_state[1]})")

            current_frame_data.append(
                go.Scatter(
                    x=line_x,
                    y=line_y,
                    mode='lines',
                    line=dict(color='blue', width=2, dash='solid'),
                    name=f'Episode {episode} Path Lines',
                    hoverinfo='text',
                    hovertext=hover_lines,
                    showlegend=False
                )
            )

            start_y, start_x = np.where(maze == 2)
            end_y, end_x = np.where(maze == 3)

            if start_y.size > 0:
                current_frame_data.append(go.Scatter(
                    x=[start_x[0]], y=[start_y[0]], mode='markers',
                    marker=dict(symbol='star', size=15, color='green', line=dict(width=1, color='black')),
                    name='Start', hoverinfo='name', showlegend=False
                ))

            if end_y.size > 0:
                current_frame_data.append(go.Scatter(
                    x=[end_x[0]], y=[end_y[0]], mode='markers',
                    marker=dict(symbol='star', size=15, color='red', line=dict(width=1, color='black')),
                    name='End', hoverinfo='name', showlegend=False
                ))

            frames.append(go.Frame(data=current_frame_data, name=str(episode), layout=go.Layout(
                title=f"Episode {episode} Trajectory",
                xaxis=dict(
                    title="X-coordinate", tickmode='array', tickvals=np.arange(max_cols),
                    ticktext=[str(j) for j in np.arange(max_cols)], range=[-0.5, max_cols - 0.5],
                    showgrid=True, zeroline=False
                ),
                yaxis=dict(
                    title="Y-coordinate", tickmode='array', tickvals=np.arange(max_rows),
                    ticktext=[str(j) for j in np.arange(max_rows)], autorange='reversed',
                    range=[max_rows - 0.5, -0.5], showgrid=True, zeroline=False,
                    scaleanchor="x", scaleratio=1
                ),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=40, r=40, t=40, b=40)
            )))

            step = dict(
                method="animate",
                args=[[f'{episode}'], dict(mode="immediate", frame=dict(duration=100, redraw=True), transition=dict(duration=0))],
                label=str(episode)
            )
            steps.append(step)

        initial_episode = sorted_episodes[0]
        initial_maze_obj, initial_start_location = experiment_result.maze_history[initial_episode]
        initial_maze = initial_maze_obj.maze
        initial_trajectory = experiment_result.trajectory_history[initial_episode]

        initial_data = []

        initial_data.append(
            go.Heatmap(
                z=initial_maze,
                colorscale=[[0.0, 'white'], [0.333, 'gray'], [0.666, 'green'], [1.0, 'red']],
                zmin=0, zmax=3, showscale=False, name='Maze Grid'
            )
        )

        initial_line_x = []
        initial_line_y = []
        initial_hover_lines = []
        for j, (current_state, next_state) in enumerate(initial_trajectory):
            initial_line_x.extend([current_state[1], next_state[1], None])
            initial_line_y.extend([current_state[0], next_state[0], None])
            initial_hover_lines.append(f"Episode: {initial_episode}<br>Step: {j}<br>Transition: ({current_state[0]},{current_state[1]}) -> ({next_state[0]},{next_state[1]})")

        initial_data.append(
            go.Scatter(
                x=initial_line_x,
                y=initial_line_y,
                mode='lines',
                line=dict(color='blue', width=2, dash='solid'),
                name=f'Episode {initial_episode} Path Lines',
                hoverinfo='text',
                hovertext=initial_hover_lines,
                showlegend=False
            )
        )

        initial_start_y, initial_start_x = np.where(initial_maze == 2)
        initial_end_y, initial_end_x = np.where(initial_maze == 3)

        if initial_start_y.size > 0:
            initial_data.append(go.Scatter(
                x=[initial_start_x[0]], y=[initial_start_y[0]], mode='markers',
                marker=dict(symbol='star', size=15, color='green', line=dict(width=1, color='black')),
                name='Start', hoverinfo='name', showlegend=False
            ))
        if initial_end_y.size > 0:
            initial_data.append(go.Scatter(
                x=[initial_end_x[0]], y=[initial_end_y[0]], mode='markers',
                marker=dict(symbol='star', size=15, color='red', line=dict(width=1, color='black')),
                name='End', hoverinfo='name', showlegend=False
            ))

        fig.add_traces(initial_data)
        fig.frames = frames

        fig.update_layout(
            title=f"Episode {initial_episode} Trajectory",
            xaxis=dict(
                title="X-coordinate", tickmode='array', tickvals=np.arange(max_cols),
                ticktext=[str(j) for j in np.arange(max_cols)], range=[-0.5, max_cols - 0.5],
                showgrid=True, zeroline=False
            ),
            yaxis=dict(
                title="Y-coordinate", tickmode='array', tickvals=np.arange(max_rows),
                ticktext=[str(j) for j in np.arange(max_rows)], autorange='reversed',
                range=[max_rows - 0.5, -0.5], showgrid=True, zeroline=False,
                scaleanchor="x", scaleratio=1
            ),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=40, t=40, b=40),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(label="Play",
                             method="animate",
                             args=[None, {"frame": {"duration": 100, "redraw": True},
                                           "fromcurrent": True, "transition": {"duration": 0}}]),
                        dict(label="Pause",
                             method="animate",
                             args=[[None], {"mode": "pause"}]),
                        dict(label="Stop",
                             method="animate",
                             args=[[f'{sorted_episodes[0]}'], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}])
                    ]
                )
            ],
            sliders=[dict(
                active=0,
                steps=steps,
                pad={"t": 50},
                x=0.05,
                len=0.9
            )]
        )
        return fig

    def create_averaged_trajectory_view(self, experiment_result: ExperimentResult, include_maze: bool) -> go.Figure:
        """
        Creates a Plotly figure visualizing averaged trajectories for different maze configurations.
        Includes a slider to select between unique maze configurations.
        Optionally includes the maze heatmap.
        """
        fig = go.Figure()
        frames = []
        steps = []

        maze_configs = defaultdict(list)
        unique_maze_keys = []
        maze_data_map = {}

        for episode, (maze_obj, start_loc) in experiment_result.maze_history.items():
            maze_hash = hashlib.md5(maze_obj.maze.tobytes()).hexdigest()
            config_key = (maze_hash, start_loc)

            if config_key not in unique_maze_keys:
                unique_maze_keys.append(config_key)
                maze_data_map[config_key] = (maze_obj.maze, start_loc)

            maze_configs[config_key].append(episode)

        if not unique_maze_keys:
            fig.update_layout(title="No maze configurations available to display.",
                              xaxis_visible=False, yaxis_visible=False)
            return fig

        max_rows = 0
        max_cols = 0
        for config_key in unique_maze_keys:
            maze_data, _ = maze_data_map[config_key]
            rows, cols = maze_data.shape
            max_rows = max(max_rows, rows)
            max_cols = max(max_cols, cols)

        for i, config_key in enumerate(unique_maze_keys):
            maze_data, start_location = maze_data_map[config_key]
            associated_episodes = maze_configs[config_key]

            aggregated_transitions = Counter()
            for episode in associated_episodes:
                if episode in experiment_result.trajectory_history:
                    for current_state, next_state in experiment_result.trajectory_history[episode]:
                        aggregated_transitions[(current_state, next_state)] += 1

            current_frame_data = []

            if include_maze:
                current_frame_data.append(
                    go.Heatmap(
                        z=maze_data,
                        colorscale=[
                            [0.0, 'white'], [0.333, 'gray'], [0.666, 'green'], [1.0, 'red']
                        ],
                        zmin=0, zmax=3, showscale=False, name='Maze Grid'
                    )
                )

            max_count = max(aggregated_transitions.values()) if aggregated_transitions else 1
            colorscale_plasma = px.colors.sequential.Plasma

            for (current_y, current_x), (next_y, next_x) in aggregated_transitions:
                count = aggregated_transitions[((current_y, current_x), (next_y, next_x))]
                color_idx = min(int((count / max_count) * (len(colorscale_plasma) - 1)), len(colorscale_plasma) - 1)
                edge_color = colorscale_plasma[color_idx]
                line_width = 1 + (count / max_count) * 5

                current_frame_data.append(
                    go.Scatter(
                        x=[current_x, next_x],
                        y=[current_y, next_y],
                        mode='lines',
                        line=dict(color=edge_color, width=line_width),
                        hoverinfo='text',
                        hovertext=f"Transition: ({current_y},{current_x}) -> ({next_y},{next_x})<br>Frequency: {count}",
                        showlegend=False
                    )
                )

            start_y, start_x = np.where(maze_data == 2)
            end_y, end_x = np.where(maze_data == 3)

            if start_y.size > 0:
                current_frame_data.append(go.Scatter(
                    x=[start_x[0]], y=[start_y[0]], mode='markers',
                    marker=dict(symbol='star', size=15, color='green', line=dict(width=1, color='black')),
                    name='Start', hoverinfo='name', showlegend=False
                ))

            if end_y.size > 0:
                current_frame_data.append(go.Scatter(
                    x=[end_x[0]], y=[end_y[0]], mode='markers',
                    marker=dict(symbol='star', size=15, color='red', line=dict(width=1, color='black')),
                    name='End', hoverinfo='name', showlegend=False
                ))

            frames.append(go.Frame(data=current_frame_data, name=str(i), layout=go.Layout(
                title=f"Averaged Trajectory for Maze Config {i+1} (Episodes: {len(associated_episodes)})",
                xaxis=dict(
                    title="X-coordinate", tickmode='array', tickvals=np.arange(max_cols),
                    ticktext=[str(j) for j in np.arange(max_cols)], range=[-0.5, max_cols - 0.5],
                    showgrid=True, zeroline=False
                ),
                yaxis=dict(
                    title="Y-coordinate", tickmode='array', tickvals=np.arange(max_rows),
                    ticktext=[str(j) for j in np.arange(max_rows)], autorange='reversed',
                    range=[max_rows - 0.5, -0.5], showgrid=True, zeroline=False,
                    scaleanchor="x", scaleratio=1
                ),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=40, r=40, t=40, b=40)
            )))

            step_label = f"Maze {i+1}"
            step = dict(
                method="animate",
                args=[[f'{i}'], dict(mode="immediate", frame=dict(duration=100, redraw=True), transition=dict(duration=0))],
                label=step_label
            )
            steps.append(step)

        initial_config_key = unique_maze_keys[0]
        initial_maze_data, initial_start_location = maze_data_map[initial_config_key]
        initial_associated_episodes = maze_configs[initial_config_key]

        initial_aggregated_transitions = Counter()
        for episode in initial_associated_episodes:
            if episode in experiment_result.trajectory_history:
                for current_state, next_state in experiment_result.trajectory_history[episode]:
                    initial_aggregated_transitions[(current_state, next_state)] += 1

        initial_data = []

        if include_maze:
            initial_data.append(
                go.Heatmap(
                    z=initial_maze_data,
                    colorscale=[[0.0, 'white'], [0.333, 'gray'], [0.666, 'green'], [1.0, 'red']],
                    zmin=0, zmax=3, showscale=False, name='Maze Grid'
                )
            )

        max_count = max(initial_aggregated_transitions.values()) if initial_aggregated_transitions else 1
        colorscale_plasma = px.colors.sequential.Plasma

        for (current_y, current_x), (next_y, next_x) in initial_aggregated_transitions:
            count = initial_aggregated_transitions[((current_y, current_x), (next_y, next_x))]
            color_idx = min(int((count / max_count) * (len(colorscale_plasma) - 1)), len(colorscale_plasma) - 1)
            edge_color = colorscale_plasma[color_idx]
            line_width = 1 + (count / max_count) * 5

            initial_data.append(
                go.Scatter(
                    x=[current_x, next_x],
                    y=[current_y, next_y],
                    mode='lines',
                    line=dict(color=edge_color, width=line_width),
                    hoverinfo='text',
                    hovertext=f"Transition: ({current_y},{current_x}) -> ({next_y},{next_x})<br>Frequency: {count}",
                    showlegend=False
                )
            )

        initial_start_y, initial_start_x = np.where(initial_maze_data == 2)
        initial_end_y, initial_end_x = np.where(initial_maze_data == 3)

        if initial_start_y.size > 0:
            initial_data.append(go.Scatter(
                x=[initial_start_x[0]], y=[initial_start_y[0]], mode='markers',
                marker=dict(symbol='star', size=15, color='green', line=dict(width=1, color='black')),
                name='Start', hoverinfo='name', showlegend=False
            ))
        if initial_end_y.size > 0:
            initial_data.append(go.Scatter(
                x=[end_x[0]], y=[end_y[0]], mode='markers',
                marker=dict(symbol='star', size=15, color='red', line=dict(width=1, color='black')),
                name='End', hoverinfo='name', showlegend=False
            ))

        fig.add_traces(initial_data)
        fig.frames = frames

        title_prefix = "Averaged Trajectory" if include_maze else "Averaged Trajectory (No Maze)"
        fig.update_layout(
            title=f"{title_prefix} for Maze Config 1 (Episodes: {len(initial_associated_episodes)})",
            xaxis=dict(
                title="X-coordinate", tickmode='array', tickvals=np.arange(max_cols),
                ticktext=[str(j) for j in np.arange(max_cols)], range=[-0.5, max_cols - 0.5],
                showgrid=True, zeroline=False
            ),
            yaxis=dict(
                title="Y-coordinate", tickmode='array', tickvals=np.arange(max_rows),
                ticktext=[str(j) for j in np.arange(max_rows)], autorange='reversed',
                range=[max_rows - 0.5, -0.5], showgrid=True, zeroline=False,
                scaleanchor="x", scaleratio=1
            ),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=40, t=40, b=40),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(label="Play",
                             method="animate",
                             args=[None, {"frame": {"duration": 100, "redraw": True},
                                           "fromcurrent": True, "transition": {"duration": 0}}]),
                        dict(label="Pause",
                             method="animate",
                             args=[[None], {"mode": "pause"}]),
                        dict(label="Stop",
                             method="animate",
                             args=[[f'{0}'], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}])
                    ]
                )
            ],
            sliders=[dict(
                active=0,
                steps=steps,
                pad={"t": 50},
                x=0.05,
                len=0.9
            )]
        )
        return fig

    def create_cumulative_rewards_figure(self, experiment_result: ExperimentResult) -> go.Figure:
        """
        Creates a Plotly figure visualizing the cumulative rewards over episodes.
        """
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

    def create_dashboard(self, experiment_result: ExperimentResult, experiment: Experiment) -> dash.Dash:
        """
        Creates a Dash application dashboard to visualize experiment results.

        Args:
            experiment_result (ExperimentResult): An object containing all experiment data.
            experiment (Experiment): An object containing experiment configuration details.

        Returns:
            dash.Dash: The configured Dash application.
        """
        # Define external stylesheets and scripts
        external_stylesheets = [dbc.themes.BOOTSTRAP]
        external_scripts = ['https://cdn.plot.ly/plotly-latest.min.js'] # Explicitly include Plotly.js

        app = dash.Dash(__name__, external_stylesheets=external_stylesheets, external_scripts=external_scripts)

        # Prepare hyperparameters for display, excluding None values
        hyperparams_to_display = {
            name: value for name, value in experiment.hyperparameters.model_dump().items()
            if value is not None
        }

        app.layout = dbc.Container(fluid=True, children=[
            html.H1("Experiment Results Dashboard", className="my-4 text-center"),

            dbc.Row(className="mb-4", children=[
                dbc.Col(
                    dbc.Tabs(id="visualization-tabs", active_tab="tab-episode-view", children=[
                        dbc.Tab(label="Episode Trajectory", tab_id="tab-episode-view", children=[
                            dcc.Graph(
                                id='episode-trajectory-graph',
                                figure=self.create_episode_view_with_slider(experiment_result),
                                style={'height': '600px'}
                            )
                        ]),
                        dbc.Tab(label="Averaged Trajectory (with Maze)", tab_id="tab-averaged-with-maze", children=[
                            dcc.Graph(
                                id='averaged-trajectory-with-maze-graph',
                                figure=self.create_averaged_trajectory_view(experiment_result, include_maze=True),
                                style={'height': '600px'}
                            )
                        ]),
                        dbc.Tab(label="Averaged Trajectory (no Maze)", tab_id="tab-averaged-no-maze", children=[
                            dcc.Graph(
                                id='averaged-trajectory-no-maze-graph',
                                figure=self.create_averaged_trajectory_view(experiment_result, include_maze=False),
                                style={'height': '600px'}
                            )
                        ]),
                    ]),
                    lg=8 # Visualization graphs take 2/3 of the width on large screens
                ),
                dbc.Col(
                    html.Div(children=[
                        html.Div(className="card shadow-sm p-3 mb-4", children=[
                            html.H4("Experiment Metrics", className="card-title text-center mb-3"),
                            dbc.Row(children=[
                                dbc.Col(dbc.Card(dbc.CardBody([
                                    html.H5("Success Rate", className="card-subtitle mb-2 text-muted"),
                                    html.H3(f"{experiment_result.success_rate:.2f}", className="card-text text-center"),
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
                            ], justify="center")
                        ]),
                        html.Div(className="card shadow-sm p-3 mb-4", children=[
                            html.H4("Experiment Details", className="card-title text-center mb-3"),
                            html.P(f"Experiment Name: {experiment.experiment_name}", className="card-text"),
                            html.P(f"Agent Type: {experiment.agent_type.value}", className="card-text"),
                            dbc.Accordion(
                                [
                                    dbc.AccordionItem(
                                        html.Ul([
                                            html.Li(f"{name.replace('_', ' ').title()}: {value}")
                                            for name, value in hyperparams_to_display.items()
                                        ], className="list-unstyled"),
                                        title="Hyperparameters",
                                        item_id="hyperparameters-accordion-item"
                                    ),
                                ],
                                start_collapsed=True, # Start collapsed to save space
                                always_open=False,
                                className="mt-3"
                            )
                        ])
                    ]),
                    lg=4 # Metrics and Experiment Details column takes 1/3 of the width on large screens
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