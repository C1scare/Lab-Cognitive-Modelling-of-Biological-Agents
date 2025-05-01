# maze/renderer.py

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

class MazeRenderer:
    def __init__(self, agent_color='dodgerblue', goal_color='gold', show_grid=True):
        self.agent_color = to_rgba(agent_color)
        self.goal_color = to_rgba(goal_color)
        self.show_grid = show_grid

    def render(self, env, agent_position, reward_positions=None):
        maze = env.maze
        visited = env.visited if hasattr(env, 'visited') else set()

        nrows, ncols = maze.shape
        canvas = np.ones((nrows, ncols, 3))

        # Fill walls
        for r in range(nrows):
            for c in range(ncols):
                if maze[r, c] != 0:
                    canvas[r, c] = [0, 0, 0]  # black walls

        # Visited cells - light gray
        for r, c in visited:
            canvas[r, c] = [0.8, 0.8, 0.8]

        # Goal cell
        if reward_positions:
            for r, c in reward_positions:
                canvas[r, c] = self.goal_color[:3]

        # Agent position
        r, c = agent_position
        canvas[r, c] = self.agent_color[:3]

        # Show image
        plt.imshow(canvas, interpolation='none')
        if self.show_grid:
            plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)
            plt.xticks(np.arange(-.5, ncols, 1), [])
            plt.yticks(np.arange(-.5, nrows, 1), [])
            plt.gca().set_xticks(np.arange(-.5, ncols, 1), minor=True)
            plt.gca().set_yticks(np.arange(-.5, nrows, 1), minor=True)
        else:
            plt.xticks([])
            plt.yticks([])
        plt.show()
