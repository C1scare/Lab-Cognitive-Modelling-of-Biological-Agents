import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

class MazeRenderer:
    def __init__(self, agent_color='dodgerblue', goal_color='gold', show_grid=True):
        self.agent_color = to_rgba(agent_color)
        self.goal_color = to_rgba(goal_color)
        self.show_grid = show_grid
        self._img_artist = None  # For efficient rendering updates

    def render(self, env, agent_position, reward_positions=None):
        # Original rendering logic for dynamic agent movement
        self._render_maze(env.maze, agent_position, reward_positions, visited=getattr(env, 'visited', set()))

    def render_setup(self, maze, start_positions, goal_position):
        # New static rendering method for maze setup
        self._render_maze(maze, agent_position=None, reward_positions=[goal_position], start_positions=start_positions)

    def _render_maze(self, maze, agent_position=None, reward_positions=None, visited=set(), start_positions=None):
        nrows, ncols = maze.shape
        canvas = np.ones((nrows, ncols, 3))

        # Fill walls
        for r_idx in range(nrows):
            for c_idx in range(ncols):
                if maze[r_idx, c_idx] != 0:
                    canvas[r_idx, c_idx] = [0, 0, 0]  # black walls

        # Visited cells - light gray
        for r_vis, c_vis in visited:
            canvas[r_vis, c_vis] = [0.8, 0.8, 0.8]

        # Goal cell
        if reward_positions:
            for r_goal, c_goal in reward_positions:
                if 0 <= r_goal < nrows and 0 <= c_goal < ncols:
                    canvas[r_goal, c_goal] = self.goal_color[:3]

        # Agent position (if active)
        if agent_position:
            r_agent, c_agent = agent_position
            if 0 <= r_agent < nrows and 0 <= c_agent < ncols:
                canvas[r_agent, c_agent] = self.agent_color[:3]

        # Start positions
        if start_positions:
            for r_start, c_start in start_positions:
                if 0 <= r_start < nrows and 0 <= c_start < ncols:
                    canvas[r_start, c_start] = self.agent_color[:3]  # Blue by default

        ax = plt.gca()

        if self._img_artist is None or self._img_artist.axes != ax:
            self._img_artist = ax.imshow(canvas, interpolation='none')
        else:
            self._img_artist.set_data(canvas)

        self._img_artist.set_extent((-.5, ncols - 0.5, nrows - 0.5, -.5))

        if self.show_grid:
            ax.set_xticks(np.arange(-.5, ncols, 1))
            ax.set_yticks(np.arange(-.5, nrows, 1))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
