#from pydantic import BaseModel, Field
from typing import List, Tuple, Dict, Union
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

# Helper function to safely get min/max for normalization
def get_global_min_max(history_dict_or_list: Union[Dict[int, np.ndarray], List[float]]):
    all_values = []
    if isinstance(history_dict_or_list, dict):
        for episode_data in history_dict_or_list.values():
            if isinstance(episode_data, np.ndarray):
                all_values.extend(episode_data.flatten())
    elif isinstance(history_dict_or_list, list):
        all_values = history_dict_or_list # Directly use the list if it's already a list of values

    if not all_values:
        return 0, 1 # Default to [0, 1] if no data
    # Ensure min_val != max_val for heatmap scaling
    min_val = min(all_values)
    max_val = max(all_values)
    if min_val == max_val:
        # If all values are the same, expand the range slightly to prevent issues
        # Adjust min_val to be slightly smaller and max_val slightly larger, unless it's zero
        if min_val == 0:
            max_val = 1.0 # default to 0-1 for zero values
        else:
            min_val = min_val * 0.9
            max_val = max_val * 1.1
        if min_val == max_val: # Final safeguard if 0.9/1.1 still make them equal (e.g., if min_val is tiny)
            min_val -= 1e-9
            max_val += 1e-9
    return min_val, max_val


class MazeVisualizer:
    """
    A class to handle the visualization of a maze and agent trajectories,
    including episode-specific views and averaged views across maze configurations.
    """

    def _create_heatmap_trace(self, data: np.ndarray,
                               maze_shape: Tuple[int, int], min_val: float, max_val: float,
                               title: str, colorscale: str) -> go.Heatmap:
        rows, cols = maze_shape
        
        return go.Heatmap(
            z=data,
            x=np.arange(cols),
            y=np.arange(rows),
            colorscale=colorscale,
            zmin=min_val,
            zmax=max_val,
            showscale=False,
            name=title
        )

    def create_episode_view_with_slider(self, experiment_result: ExperimentResult) -> go.Figure:
        """
        Creates a Plotly figure with a slider to select and display individual episode trajectories,
        along with Q-mean, curiosity, and uncertainty heatmaps for that episode.
        """
        fig = go.Figure()
        frames = []
        steps = []

        sorted_episodes = sorted(experiment_result.trajectory_history.keys())

        if not sorted_episodes:
            fig.update_layout(title="No episode data available to display.",
                              xaxis_visible=False, yaxis_visible=False)
            return fig

        max_rows = 0
        max_cols = 0
        for episode in sorted_episodes:
            if episode in experiment_result.maze_history:
                maze_obj, _, _ = experiment_result.maze_history[episode]
                rows, cols = maze_obj.maze.shape
                max_rows = max(max_rows, rows)
                max_cols = max(max_cols, cols)
            else:
                print(f"Warning: Maze history missing for episode {episode}. Skipping.")
                continue

        # Calculate global min/max for normalization across all episodes
        q_mean_min, q_mean_max = get_global_min_max(experiment_result.q_mean_history)
        curiosity_min, curiosity_max = get_global_min_max(experiment_result.curiosity_history)
        uncertainty_min, uncertainty_max = get_global_min_max(experiment_result.uncertainty_history)

        for i, episode in enumerate(sorted_episodes):
            if episode not in experiment_result.maze_history or episode not in experiment_result.trajectory_history:
                continue

            maze_obj, start_location, goal_location = experiment_result.maze_history[episode]
            maze = maze_obj.maze
            maze = maze_obj.maze.copy()
            maze[maze > 1] = 0
            trajectory = experiment_result.trajectory_history[episode]
            # Use .get with a default of an empty array with the correct shape
            q_mean_data = experiment_result.q_mean_history.get(episode, np.zeros(maze.shape))
            curiosity_data = experiment_result.curiosity_history.get(episode, np.zeros(maze.shape))
            uncertainty_data = experiment_result.uncertainty_history.get(episode, np.zeros(maze.shape))

            current_frame_data = []

            # Subplot 1: Trajectory (Maze Grid + Path + Start/End Markers)
            line_x = []
            line_y = []
            hover_lines = []
            for j, (current_state, next_state) in enumerate(trajectory):
                line_x.extend([current_state[1], next_state[1], None])
                line_y.extend([current_state[0], next_state[0], None])
                hover_lines.append(f"Episode: {episode}<br>Step: {j}<br>Transition: ({current_state[0]},{current_state[1]}) -> ({next_state[0]},{next_state[1]})")

            current_frame_data.append(
                go.Heatmap(
                    z=maze,
                    colorscale=[[0.0, 'white'], [1.0, 'gray']], # Only 0 for empty, 1 for walls
                    zmin=0, zmax=1, showscale=False, name='Maze Grid',
                    xaxis='x1', yaxis='y1'
                )
            )
            current_frame_data.append(
                go.Scatter(
                    x=line_x, y=line_y, mode='lines',
                    line=dict(color='blue', width=2, dash='solid'),
                    name=f'Episode {episode} Path Lines', hoverinfo='text', hovertext=hover_lines, showlegend=False,
                    xaxis='x1', yaxis='y1'
                )
            )
            
            # Use start_location and goal_location directly
            start_y, start_x = start_location
            end_y, end_x = goal_location

            current_frame_data.append(go.Scatter(
                x=[start_x], y=[start_y], mode='markers',
                marker=dict(symbol='star', size=15, color='green', line=dict(width=1, color='black')),
                name='Start', hoverinfo='name', showlegend=False,
                xaxis='x1', yaxis='y1'
            ))
            current_frame_data.append(go.Scatter(
                x=[end_x], y=[end_y], mode='markers',
                marker=dict(symbol='star', size=15, color='red', line=dict(width=1, color='black')),
                name='End', hoverinfo='name', showlegend=False,
                xaxis='x1', yaxis='y1'
            ))

            # Subplot 2: Q-Mean Heatmap
            current_frame_data.append(self._create_heatmap_trace(
                q_mean_data, maze.shape, q_mean_min, q_mean_max, 'Q-Mean', 'Viridis'
            ).update(xaxis='x2', yaxis='y2'))

            # Subplot 3: Curiosity Heatmap
            current_frame_data.append(self._create_heatmap_trace(
                curiosity_data, maze.shape, curiosity_min, curiosity_max, 'Curiosity', 'Plasma'
            ).update(xaxis='x3', yaxis='y3'))

            # Subplot 4: Uncertainty Heatmap
            current_frame_data.append(self._create_heatmap_trace(
                uncertainty_data, maze.shape, uncertainty_min, uncertainty_max, 'Uncertainty', 'Cividis'
            ).update(xaxis='x4', yaxis='y4'))

            frames.append(go.Frame(data=current_frame_data, name=str(episode), layout=go.Layout(
                title=f"Episode {episode} Trajectory & Metrics",
                grid=dict(rows=2, columns=2, pattern="independent"),
                xaxis1=dict(title="X", range=[-0.5, max_cols - 0.5], showgrid=True, zeroline=False, scaleanchor="y1", scaleratio=1),
                yaxis1=dict(title="Y", range=[max_rows - 0.5, -0.5], showgrid=True, zeroline=False, scaleanchor="x1", scaleratio=1),
                xaxis2=dict(title="X", range=[-0.5, max_cols - 0.5], showgrid=True, zeroline=False, scaleanchor="y2", scaleratio=1),
                yaxis2=dict(title="Y", range=[max_rows - 0.5, -0.5], showgrid=True, zeroline=False, scaleanchor="x2", scaleratio=1),
                xaxis3=dict(title="X", range=[-0.5, max_cols - 0.5], showgrid=True, zeroline=False, scaleanchor="y3", scaleratio=1),
                yaxis3=dict(title="Y", range=[max_rows - 0.5, -0.5], showgrid=True, zeroline=False, scaleanchor="x3", scaleratio=1),
                xaxis4=dict(title="X", range=[-0.5, max_cols - 0.5], showgrid=True, zeroline=False, scaleanchor="y4", scaleratio=1),
                yaxis4=dict(title="Y", range=[max_rows - 0.5, -0.5], showgrid=True, zeroline=False, scaleanchor="x4", scaleratio=1),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=40, r=40, t=100, b=40)
            )))

            step = dict(
                method="animate",
                args=[[f'{episode}'], dict(mode="immediate", frame=dict(duration=300, redraw=True), transition=dict(duration=100))],
                label=str(episode)
            )
            steps.append(step)

        # Initial frame setup
        initial_episode = sorted_episodes[0]
        initial_maze_obj, initial_start_location, initial_goal_location = experiment_result.maze_history[initial_episode]
        initial_maze = initial_maze_obj.maze.copy()
        initial_maze[initial_maze > 1] = 0
        initial_trajectory = experiment_result.trajectory_history[initial_episode]
        initial_q_mean_data = experiment_result.q_mean_history.get(initial_episode, np.zeros(initial_maze.shape))
        initial_curiosity_data = experiment_result.curiosity_history.get(initial_episode, np.zeros(initial_maze.shape))
        initial_uncertainty_data = experiment_result.uncertainty_history.get(initial_episode, np.zeros(initial_maze.shape))

        initial_data = []

        # Initial Trajectory subplot
        initial_line_x = []
        initial_line_y = []
        initial_hover_lines = []
        for j, (current_state, next_state) in enumerate(initial_trajectory):
            initial_line_x.extend([current_state[1], next_state[1], None])
            initial_line_y.extend([current_state[0], next_state[0], None])
            initial_hover_lines.append(f"Episode: {initial_episode}<br>Step: {j}<br>Transition: ({current_state[0]},{current_state[1]}) -> ({next_state[0]},{next_state[1]})")

        initial_data.append(
            go.Heatmap(
                z=initial_maze,
                colorscale=[[0.0, 'white'], [1.0, 'gray']],
                zmin=0, zmax=1, showscale=False, name='Maze Grid',
                xaxis='x1', yaxis='y1'
            )
        )
        initial_data.append(
            go.Scatter(
                x=initial_line_x, y=initial_line_y, mode='lines',
                line=dict(color='blue', width=2, dash='solid'),
                name=f'Episode {initial_episode} Path Lines', hoverinfo='text', hovertext=initial_hover_lines, showlegend=False,
                xaxis='x1', yaxis='y1'
            )
        )
        initial_start_y, initial_start_x = initial_start_location
        initial_end_y, initial_end_x = initial_goal_location
        initial_data.append(go.Scatter(
            x=[initial_start_x], y=[initial_start_y], mode='markers',
            marker=dict(symbol='star', size=15, color='green', line=dict(width=1, color='black')),
            name='Start', hoverinfo='name', showlegend=False,
            xaxis='x1', yaxis='y1'
        ))
        initial_data.append(go.Scatter(
            x=[initial_end_x], y=[initial_end_y], mode='markers',
            marker=dict(symbol='star', size=15, color='red', line=dict(width=1, color='black')),
            name='End', hoverinfo='name', showlegend=False,
            xaxis='x1', yaxis='y1'
        ))

        # Initial Q-Mean Heatmap
        initial_data.append(self._create_heatmap_trace(
            initial_q_mean_data, initial_maze.shape, q_mean_min, q_mean_max, 'Q-Mean', 'Viridis'
        ).update(xaxis='x2', yaxis='y2'))

        # Initial Curiosity Heatmap
        initial_data.append(self._create_heatmap_trace(
            initial_curiosity_data, initial_maze.shape, curiosity_min, curiosity_max, 'Curiosity', 'Plasma'
        ).update(xaxis='x3', yaxis='y3'))

        # Initial Uncertainty Heatmap
        initial_data.append(self._create_heatmap_trace(
            initial_uncertainty_data, initial_maze.shape, uncertainty_min, uncertainty_max, 'Uncertainty', 'Cividis'
        ).update(xaxis='x4', yaxis='y4'))

        fig.add_traces(initial_data)
        fig.frames = frames

        fig.update_layout(
            title=f"Episode {initial_episode} Trajectory & Metrics",
            grid=dict(rows=2, columns=2, pattern="independent"),
            xaxis1=dict(title="X", tickmode='array', tickvals=np.arange(max_cols), ticktext=[str(j) for j in np.arange(max_cols)], range=[-0.5, max_cols - 0.5], showgrid=True, zeroline=False, scaleanchor="y1", scaleratio=1),
            yaxis1=dict(title="Y", tickmode='array', tickvals=np.arange(max_rows), ticktext=[str(j) for j in np.arange(max_rows)], autorange='reversed', range=[max_rows - 0.5, -0.5], showgrid=True, zeroline=False, scaleanchor="x1", scaleratio=1),
            xaxis2=dict(title="X", tickmode='array', tickvals=np.arange(max_cols), ticktext=[str(j) for j in np.arange(max_cols)], range=[-0.5, max_cols - 0.5], showgrid=True, zeroline=False, scaleanchor="y2", scaleratio=1),
            yaxis2=dict(title="Y", tickmode='array', tickvals=np.arange(max_rows), ticktext=[str(j) for j in np.arange(max_rows)], autorange='reversed', range=[max_rows - 0.5, -0.5], showgrid=True, zeroline=False, scaleanchor="x2", scaleratio=1),
            xaxis3=dict(title="X", tickmode='array', tickvals=np.arange(max_cols), ticktext=[str(j) for j in np.arange(max_cols)], range=[-0.5, max_cols - 0.5], showgrid=True, zeroline=False, scaleanchor="y3", scaleratio=1),
            yaxis3=dict(title="Y", tickmode='array', tickvals=np.arange(max_rows), ticktext=[str(j) for j in np.arange(max_rows)], autorange='reversed', range=[max_rows - 0.5, -0.5], showgrid=True, zeroline=False, scaleanchor="x3", scaleratio=1),
            xaxis4=dict(title="X", tickmode='array', tickvals=np.arange(max_cols), ticktext=[str(j) for j in np.arange(max_cols)], range=[-0.5, max_cols - 0.5], showgrid=True, zeroline=False, scaleanchor="y4", scaleratio=1),
            yaxis4=dict(title="Y", tickmode='array', tickvals=np.arange(max_rows), ticktext=[str(j) for j in np.arange(max_rows)], autorange='reversed', range=[max_rows - 0.5, -0.5], showgrid=True, zeroline=False, scaleanchor="x4", scaleratio=1),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=40, t=100, b=40),
            height=800, # Explicitly set height for the figure
            width=800,  # Explicitly set width for the figure to make it square
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(label="Play",
                             method="animate",
                             args=[None, {"frame": {"duration": 300, "redraw": True},
                                          "fromcurrent": True, "transition": {"duration": 300}}]),
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
        Creates a Plotly figure visualizing averaged trajectories and averaged Q-mean, curiosity,
        and uncertainty heatmaps for different maze configurations. Includes a slider to select
        between unique maze configurations. Optionally includes the maze heatmap.
        """
        fig = go.Figure()
        frames = []
        steps = []

        maze_configs = defaultdict(list)
        unique_maze_keys = []
        maze_data_map = {}

        for episode, (maze_obj, start_loc, goal_loc) in experiment_result.maze_history.items():
            maze_hash = hashlib.md5(maze_obj.maze.tobytes()).hexdigest()
            config_key = (maze_hash, start_loc, goal_loc) # Include goal_loc in config_key

            if config_key not in unique_maze_keys:
                unique_maze_keys.append(config_key)
                maze_data_map[config_key] = (maze_obj.maze, start_loc, goal_loc)

            maze_configs[config_key].append(episode)

        if not unique_maze_keys:
            fig.update_layout(title="No maze configurations available to display.",
                              xaxis_visible=False, yaxis_visible=False)
            return fig

        max_rows = 0
        max_cols = 0
        for config_key in unique_maze_keys:
            maze_data, _, _ = maze_data_map[config_key]
            rows, cols = maze_data.shape
            max_rows = max(max_rows, rows)
            max_cols = max(max_cols, cols)

        # Calculate global min/max for normalization across all configurations
        all_q_mean_values = []
        all_curiosity_values = []
        all_uncertainty_values = []

        for config_key in unique_maze_keys:
            associated_episodes = maze_configs[config_key]
            for episode in associated_episodes:
                if episode in experiment_result.q_mean_history:
                    all_q_mean_values.extend(experiment_result.q_mean_history[episode].flatten())
                if episode in experiment_result.curiosity_history:
                    all_curiosity_values.extend(experiment_result.curiosity_history[episode].flatten())
                if episode in experiment_result.uncertainty_history:
                    all_uncertainty_values.extend(experiment_result.uncertainty_history[episode].flatten())
        
        q_mean_min, q_mean_max = get_global_min_max(all_q_mean_values)
        curiosity_min, curiosity_max = get_global_min_max(all_curiosity_values)
        uncertainty_min, uncertainty_max = get_global_min_max(all_uncertainty_values)


        for i, config_key in enumerate(unique_maze_keys):
            maze_data, start_location, goal_location = maze_data_map[config_key]
            associated_episodes = maze_configs[config_key]

            aggregated_transitions = Counter()
            aggregated_q_mean_arrays = []
            aggregated_curiosity_arrays = []
            aggregated_uncertainty_arrays = []

            for episode in associated_episodes:
                if episode in experiment_result.trajectory_history:
                    for current_state, next_state in experiment_result.trajectory_history[episode]:
                        aggregated_transitions[(current_state, next_state)] += 1
                
                # Append the full heatmap arrays to aggregate them
                if episode in experiment_result.q_mean_history:
                    aggregated_q_mean_arrays.append(experiment_result.q_mean_history[episode])
                if episode in experiment_result.curiosity_history:
                    aggregated_curiosity_arrays.append(experiment_result.curiosity_history[episode])
                if episode in experiment_result.uncertainty_history:
                    aggregated_uncertainty_arrays.append(experiment_result.uncertainty_history[episode])
            
            # Average the aggregated metric arrays
            averaged_q_mean = np.mean(aggregated_q_mean_arrays, axis=0) if len(aggregated_q_mean_arrays) > 0 else np.zeros(maze_data.shape)
            averaged_curiosity = np.mean(aggregated_curiosity_arrays, axis=0) if len(aggregated_curiosity_arrays) > 0 else np.zeros(maze_data.shape)
            averaged_uncertainty = np.mean(aggregated_uncertainty_arrays, axis=0) if len(aggregated_uncertainty_arrays) > 0 else np.zeros(maze_data.shape)

            current_frame_data = []

            # Subplot 1: Averaged Trajectory (Maze Grid + Frequncy lines + Start/End Markers)
            if include_maze:
                current_frame_data.append(
                    go.Heatmap(
                        z=maze_data,
                        colorscale=[[0.0, 'white'], [1.0, 'gray']],
                        zmin=0, zmax=1, showscale=False, name='Maze Grid',
                        xaxis='x1', yaxis='y1'
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
                        x=[current_x, next_x], y=[current_y, next_y], mode='lines', # Fixed: changed next_x to next_y
                        line=dict(color=edge_color, width=line_width),
                        hoverinfo='text', hovertext=f"Transition: ({current_y},{current_x}) -> ({next_y},{next_x})<br>Frequency: {count}",
                        showlegend=False, xaxis='x1', yaxis='y1'
                    )
                )

            start_y, start_x = start_location
            end_y, end_x = goal_location

            current_frame_data.append(go.Scatter(
                x=[start_x], y=[start_y], mode='markers',
                marker=dict(symbol='star', size=15, color='green', line=dict(width=1, color='black')),
                name='Start', hoverinfo='name', showlegend=False,
                xaxis='x1', yaxis='y1'
            ))
            current_frame_data.append(go.Scatter(
                x=[end_x], y=[end_y], mode='markers',
                marker=dict(symbol='star', size=15, color='red', line=dict(width=1, color='black')),
                name='End', hoverinfo='name', showlegend=False,
                xaxis='x1', yaxis='y1'
            ))

            # Subplot 2: Averaged Q-Mean Heatmap
            current_frame_data.append(self._create_heatmap_trace(
                averaged_q_mean, maze_data.shape, q_mean_min, q_mean_max, 'Averaged Q-Mean', 'Viridis'
            ).update(xaxis='x2', yaxis='y2'))

            # Subplot 3: Averaged Curiosity Heatmap
            current_frame_data.append(self._create_heatmap_trace(
                averaged_curiosity, maze_data.shape, curiosity_min, curiosity_max, 'Averaged Curiosity', 'Plasma'
            ).update(xaxis='x3', yaxis='y3'))

            # Subplot 4: Averaged Uncertainty Heatmap
            current_frame_data.append(self._create_heatmap_trace(
                averaged_uncertainty, maze_data.shape, uncertainty_min, uncertainty_max, 'Averaged Uncertainty', 'Cividis'
            ).update(xaxis='x4', yaxis='y4'))

            frames.append(go.Frame(data=current_frame_data, name=str(i), layout=go.Layout(
                title=f"Averaged Trajectory & Metrics for Maze Config {i+1} (Episodes: {len(associated_episodes)})",
                grid=dict(rows=2, columns=2, pattern="independent"),
                xaxis1=dict(title="X", range=[-0.5, max_cols - 0.5], showgrid=True, zeroline=False, scaleanchor="y1", scaleratio=1),
                yaxis1=dict(title="Y", range=[max_rows - 0.5, -0.5], showgrid=True, zeroline=False, scaleanchor="x1", scaleratio=1),
                xaxis2=dict(title="X", range=[-0.5, max_cols - 0.5], showgrid=True, zeroline=False, scaleanchor="y2", scaleratio=1),
                yaxis2=dict(title="Y", range=[max_rows - 0.5, -0.5], showgrid=True, zeroline=False, scaleanchor="x2", scaleratio=1),
                xaxis3=dict(title="X", range=[-0.5, max_cols - 0.5], showgrid=True, zeroline=False, scaleanchor="y3", scaleratio=1),
                yaxis3=dict(title="Y", range=[max_rows - 0.5, -0.5], showgrid=True, zeroline=False, scaleanchor="x3", scaleratio=1),
                xaxis4=dict(title="X", range=[-0.5, max_cols - 0.5], showgrid=True, zeroline=False, scaleanchor="y4", scaleratio=1),
                yaxis4=dict(title="Y", range=[max_rows - 0.5, -0.5], showgrid=True, zeroline=False, scaleanchor="x4", scaleratio=1),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=40, r=40, t=100, b=40)
            )))

            step_label = f"Maze {i+1}"
            step = dict(
                method="animate",
                args=[[f'{i}'], dict(mode="immediate", frame=dict(duration=300, redraw=True), transition=dict(duration=100))],
                label=step_label
            )
            steps.append(step)

        # Initial frame setup for averaged view
        initial_config_key = unique_maze_keys[0]
        initial_maze_data, initial_start_location, initial_goal_location = maze_data_map[initial_config_key]
        initial_associated_episodes = maze_configs[initial_config_key]

        initial_aggregated_transitions = Counter()
        initial_aggregated_q_mean_arrays = []
        initial_aggregated_curiosity_arrays = []
        initial_aggregated_uncertainty_arrays = []

        for episode in initial_associated_episodes:
            if episode in experiment_result.trajectory_history:
                for current_state, next_state in experiment_result.trajectory_history[episode]:
                    initial_aggregated_transitions[(current_state, next_state)] += 1
            if episode in experiment_result.q_mean_history:
                initial_aggregated_q_mean_arrays.append(experiment_result.q_mean_history[episode])
            if episode in experiment_result.curiosity_history:
                initial_aggregated_curiosity_arrays.append(experiment_result.curiosity_history[episode])
            if episode in experiment_result.uncertainty_history:
                initial_aggregated_uncertainty_arrays.append(experiment_result.uncertainty_history[episode])
        
        initial_averaged_q_mean = np.mean(initial_aggregated_q_mean_arrays, axis=0) if len(initial_aggregated_q_mean_arrays) > 0 else np.zeros(initial_maze_data.shape)
        initial_averaged_curiosity = np.mean(initial_aggregated_curiosity_arrays, axis=0) if len(initial_aggregated_curiosity_arrays) > 0 else np.zeros(initial_maze_data.shape)
        initial_averaged_uncertainty = np.mean(initial_aggregated_uncertainty_arrays, axis=0) if len(initial_aggregated_uncertainty_arrays) > 0 else np.zeros(initial_maze_data.shape)

        initial_data = []

        if include_maze:
            initial_data.append(
                go.Heatmap(
                    z=initial_maze_data,
                    colorscale=[[0.0, 'white'], [1.0, 'gray']], # Only 0 for empty, 1 for walls
                    zmin=0, zmax=1, showscale=False, name='Maze Grid',
                    xaxis='x1', yaxis='y1'
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
                    x=[current_x, next_x], y=[current_y, next_y], mode='lines',
                    line=dict(color=edge_color, width=line_width),
                    hoverinfo='text', hovertext=f"Transition: ({current_y},{current_x}) -> ({next_y},{next_x})<br>Frequency: {count}",
                    showlegend=False, xaxis='x1', yaxis='y1'
                )
            )

        initial_start_y, initial_start_x = initial_start_location
        initial_end_y, initial_end_x = initial_goal_location

        initial_data.append(go.Scatter(
            x=[initial_start_x], y=[initial_start_y], mode='markers',
            marker=dict(symbol='star', size=15, color='green', line=dict(width=1, color='black')),
            name='Start', hoverinfo='name', showlegend=False,
            xaxis='x1', yaxis='y1'
        ))
        initial_data.append(go.Scatter(
            x=[initial_end_x], y=[initial_end_y], mode='markers',
            marker=dict(symbol='star', size=15, color='red', line=dict(width=1, color='black')),
            name='End', hoverinfo='name', showlegend=False,
            xaxis='x1', yaxis='y1'
        ))

        initial_data.append(self._create_heatmap_trace(
            initial_averaged_q_mean, initial_maze_data.shape, q_mean_min, q_mean_max, 'Averaged Q-Mean', 'Viridis'
        ).update(xaxis='x2', yaxis='y2'))

        initial_data.append(self._create_heatmap_trace(
            initial_averaged_curiosity, initial_maze_data.shape, curiosity_min, curiosity_max, 'Averaged Curiosity', 'Plasma'
        ).update(xaxis='x3', yaxis='y3'))

        initial_data.append(self._create_heatmap_trace(
            initial_averaged_uncertainty, initial_maze_data.shape, uncertainty_min, uncertainty_max, 'Averaged Uncertainty', 'Cividis'
        ).update(xaxis='x4', yaxis='y4'))

        fig.add_traces(initial_data)
        fig.frames = frames

        title_prefix = "Averaged Trajectory" if include_maze else "Averaged Trajectory (No Maze)"
        fig.update_layout(
            title=f"{title_prefix} & Metrics for Maze Config 1 (Episodes: {len(initial_associated_episodes)})",
            grid=dict(rows=2, columns=2, pattern="independent"),
            xaxis1=dict(title="X", tickmode='array', tickvals=np.arange(max_cols), ticktext=[str(j) for j in np.arange(max_cols)], range=[-0.5, max_cols - 0.5], showgrid=True, zeroline=False, scaleanchor="y1", scaleratio=1),
            yaxis1=dict(title="Y", tickmode='array', tickvals=np.arange(max_rows), ticktext=[str(j) for j in np.arange(max_rows)], autorange='reversed', range=[max_rows - 0.5, -0.5], showgrid=True, zeroline=False, scaleanchor="x1", scaleratio=1),
            xaxis2=dict(title="X", tickmode='array', tickvals=np.arange(max_cols), ticktext=[str(j) for j in np.arange(max_cols)], range=[-0.5, max_cols - 0.5], showgrid=True, zeroline=False, scaleanchor="y2", scaleratio=1),
            yaxis2=dict(title="Y", tickmode='array', tickvals=np.arange(max_rows), ticktext=[str(j) for j in np.arange(max_rows)], autorange='reversed', range=[max_rows - 0.5, -0.5], showgrid=True, zeroline=False, scaleanchor="x2", scaleratio=1),
            xaxis3=dict(title="X", tickmode='array', tickvals=np.arange(max_cols), ticktext=[str(j) for j in np.arange(max_cols)], range=[-0.5, max_cols - 0.5], showgrid=True, zeroline=False, scaleanchor="y3", scaleratio=1),
            yaxis3=dict(title="Y", tickmode='array', tickvals=np.arange(max_rows), ticktext=[str(j) for j in np.arange(max_rows)], autorange='reversed', range=[max_rows - 0.5, -0.5], showgrid=True, zeroline=False, scaleanchor="x3", scaleratio=1),
            xaxis4=dict(title="X", tickmode='array', tickvals=np.arange(max_cols), ticktext=[str(j) for j in np.arange(max_cols)], range=[-0.5, max_cols - 0.5], showgrid=True, zeroline=False, scaleanchor="y4", scaleratio=1),
            yaxis4=dict(title="Y", tickmode='array', tickvals=np.arange(max_rows), ticktext=[str(j) for j in np.arange(max_rows)], autorange='reversed', range=[max_rows - 0.5, -0.5], showgrid=True, zeroline=False, scaleanchor="x4", scaleratio=1),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=40, t=100, b=40),
            height=800, # Explicitly set height for the figure
            width=800,  # Explicitly set width for the figure to make it square
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(label="Play",
                             method="animate",
                             args=[None, {"frame": {"duration": 300, "redraw": True},
                                          "fromcurrent": True, "transition": {"duration": 100}}]),
                        dict(label="Pause",
                             method="animate",
                             args=[[None], {"mode": "pause"}]),
                        dict(label="Stop",
                             method="animate",
                             args=[[f'{unique_maze_keys[0]}'], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}])
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
    
    def create_uncertainty_figure(self, experiment_result: ExperimentResult) -> go.Figure:
        """
        Creates a Plotly figure visualizing the overall uncertainty over episodes.
        """
        uncertainty = experiment_result.uncertainties
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(range(len(uncertainty))),
                y=uncertainty,
                mode='lines+markers',
                name='Uncertainty',
                line=dict(color='purple', width=2),
                marker=dict(size=4)
            )
        )
        fig.update_layout(
            title="Overall Uncertainty Over Episodes",
            xaxis_title="Episode",
            yaxis_title="Uncertainty",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=40, t=40, b=40)
        )
        return fig
    
    def create_curiosity_figure(self, experiment_result: ExperimentResult) -> go.Figure:
        """
        Creates a Plotly figure visualizing the overall curiosity over episodes.
        """
        curiosity = experiment_result.curiosity
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(range(len(curiosity))),
                y=curiosity,
                mode='lines+markers',
                name='Curiosity',
                line=dict(color='purple', width=2),
                marker=dict(size=4)
            )
        )
        fig.update_layout(
            title="Overall Curiosity Over Episodes",
            xaxis_title="Episode",
            yaxis_title="Curiosity",
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
                        dbc.Tab(label="Episode Trajectory & Metrics", tab_id="tab-episode-view", children=[
                            dcc.Graph(
                                id='episode-trajectory-graph',
                                figure=self.create_episode_view_with_slider(experiment_result),
                                # height and width are now set in the figure object itself for better control
                                style={'height': '800px'}
                            )
                        ]),
                        dbc.Tab(label="Averaged Trajectory (with Maze) & Metrics", tab_id="tab-averaged-with-maze", children=[
                            dcc.Graph(
                                id='averaged-trajectory-with-maze-graph',
                                figure=self.create_averaged_trajectory_view(experiment_result, include_maze=True),
                                # height and width are now set in the figure object itself for better control
                                style={'height': '800px'}
                            )
                        ]),
                        dbc.Tab(label="Averaged Trajectory (no Maze) & Metrics", tab_id="tab-averaged-no-maze", children=[
                            dcc.Graph(
                                id='averaged-trajectory-no-maze-graph',
                                figure=self.create_averaged_trajectory_view(experiment_result, include_maze=False),
                                # height and width are now set in the figure object itself for better control
                                style={'height': '800px'}
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
            ]),

            dbc.Row(className="mb-4", children=[
                dbc.Col(
                    dcc.Graph(
                        id='uncertainty-graph',
                        figure=self.create_uncertainty_figure(experiment_result),
                        style={'height': '400px'}
                    ),
                    width=12 # Uncertainty takes full width below
                )
            ]),

            dbc.Row(className="mb-4", children=[
                dbc.Col(
                    dcc.Graph(
                        id='curiosity-graph',
                        figure=self.create_curiosity_figure(experiment_result),
                        style={'height': '400px'}
                    ),
                    width=12 # Curiosity takes full width below
                )
            ])

        ])
        return app