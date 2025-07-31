import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

class MazeRenderer:
    """
    Renders a grid-based maze environment using matplotlib.

    This class visualizes the maze, agent, goal, and visited cells.
    It supports efficient updates for animations and optional grid display.

    Attributes:
        agent_color: RGBA color for the agent.
        goal_color: RGBA color for the goal cell.
        show_grid: Whether to display grid lines.
        _img_artist: Internal matplotlib image artist for efficient updates.
    """
    
    def __init__(self, agent_color='dodgerblue', goal_color='gold', show_grid=True):
        """
        Initialize the MazeRenderer.

        Args:
            agent_color: Color for the agent (default 'dodgerblue').
            goal_color: Color for the goal cell (default 'gold').
            show_grid: Whether to display grid lines (default True).
    """
        self.agent_color = to_rgba(agent_color)
        self.goal_color = to_rgba(goal_color)
        self.show_grid = show_grid
        self._img_artist = None  # To store the imshow artist for efficient updates

    def render(self, env, agent_position, reward_positions=None):
        """
        Render the maze, agent, goal, and visited cells using matplotlib.

        Args:
            env: Maze environment object with .maze and .visited attributes.
            agent_position: Tuple (row, col) for the agent's position.
            reward_positions: Optional list of (row, col) tuples for goal/reward cells.
        """
        maze = env.maze
        visited = env.visited if hasattr(env, 'visited') else set()

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
                if 0 <= r_goal < nrows and 0 <= c_goal < ncols: # Boundary check
                    canvas[r_goal, c_goal] = self.goal_color[:3]

        # Agent position
        r_agent, c_agent = agent_position
        if 0 <= r_agent < nrows and 0 <= c_agent < ncols: # Boundary check
            canvas[r_agent, c_agent] = self.agent_color[:3]

        # Get current axes from the active Matplotlib figure
        ax = plt.gca()

        # Show image: Update data if artist exists for the current axes, else create new
        if self._img_artist is None or self._img_artist.axes != ax:
            self._img_artist = ax.imshow(canvas, interpolation='none')
        else:
            self._img_artist.set_data(canvas)
        
        # Ensure the image extent covers the maze appropriately
        # This helps if the y-axis is inverted by default with imshow
        self._img_artist.set_extent((-.5, ncols - 0.5, nrows - 0.5, -.5))


        if self.show_grid:
            ax.set_xticks(np.arange(-.5, ncols, 1))
            ax.set_yticks(np.arange(-.5, nrows, 1))
            ax.set_xticklabels([]) # No numbers on ticks
            ax.set_yticklabels([]) # No numbers on ticks
            ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)
            # Minor ticks for grid (if major ticks are used for labels, which they are not here)
            # ax.set_xticks(np.arange(-.5, ncols, 1), minor=True)
            # ax.set_yticks(np.arange(-.5, nrows, 1), minor=True)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)