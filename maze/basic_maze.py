import logging
from enum import Enum, IntEnum
import numpy as np
from maze.maze_renderer import MazeRenderer

class CellType(IntEnum):
    EMPTY = 0
    WALL = 1
    AGENT = 2

class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class Render(Enum):
    NOTHING = 0
    TRAINING = 1
    MOVES = 2

class Status(Enum):
    SUCCESS = 0
    FAILURE = 1
    IN_PROGRESS = 2

class BasicMaze:

    actions = {
        Action.UP: (-1, 0),
        Action.DOWN: (1, 0),
        Action.LEFT: (0, -1),
        Action.RIGHT: (0, 1)
    }

    reward_goal = 10.0
    penalty_move = -0.05
    penalty_already_visited = -0.1
    penalty_impossible_move = -0.5
    minimum_reward = -5.0

    def __init__(self, maze, start_cell=(0, 0), goal_cell=None):
        self._original_maze = maze
        self.maze = np.copy(maze)
        self.start_cell = start_cell
        nrows, ncols = self.maze.shape
        self.goal_cell = goal_cell if goal_cell else (nrows - 1, ncols - 1)

        self.empty_cells = [(r, c) for r in range(nrows) for c in range(ncols) if self.maze[r][c] == CellType.EMPTY]
        if self.goal_cell in self.empty_cells:
            self.empty_cells.remove(self.goal_cell)

        if self.maze[self.goal_cell] == CellType.WALL:
            raise ValueError("Goal cell cannot be inside a wall.")

        if self.start_cell not in self.empty_cells:
            raise ValueError("The agent must start on an empty cell.")

        if self.start_cell == self.goal_cell:
            raise ValueError("The agent cannot start on the goal cell.")

        self.renderer = MazeRenderer(
            agent_color='dodgerblue',
            goal_color='gold',
            show_grid=False
        )
        self.reset(self.start_cell)

    def reset(self, start_cell):
        self.previous_position = self.agent_position = start_cell
        self.maze = np.copy(self._original_maze)
        row, col = start_cell
        self.maze[row][col] = CellType.AGENT
        self.total_reward = 0.0
        self.visited = set()
        self.visited.add(start_cell)

    def step(self, action):
        reward = self.execute(action)
        self.total_reward += reward
        status = self.status()
        state = self.state()
        return state, reward, status

    def execute(self, action):
        row, col = self.agent_position
        delta_row, delta_col = self.actions[action]
        new_row, new_col = row + delta_row, col + delta_col

        if not self.is_valid((new_row, new_col)):
            return self.penalty_impossible_move

        self.agent_position = (new_row, new_col)

        if (new_row, new_col) in self.visited:
            return self.penalty_already_visited

        self.visited.add(self.agent_position)

        if self.agent_position == self.goal_cell:
            return self.reward_goal

        return self.penalty_move

    def is_valid(self, position):
        r, c = position
        return 0 <= r < self.maze.shape[0] and 0 <= c < self.maze.shape[1] and self.maze[r][c] != CellType.WALL

    def get_shape(self):
        return self.maze.shape

    def get_reward_at(self, position):
        if position == self.goal_cell:
            return self.reward_goal
        return None

    def status(self):
        if self.agent_position == self.goal_cell:
            return Status.SUCCESS
        elif self.total_reward < self.minimum_reward:
            return Status.FAILURE
        else:
            return Status.IN_PROGRESS

    def state(self):
        return np.array([*self.agent_position])

    def render(self):
        self.renderer.render(self, self.agent_position, reward_positions=[self.goal_cell])
