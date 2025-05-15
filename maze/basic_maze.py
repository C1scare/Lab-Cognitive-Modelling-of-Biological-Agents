from enum import Enum, IntEnum
import numpy as np
from maze.maze_renderer import MazeRenderer

class CellType(IntEnum):
    EMPTY = 0
    WALL = 1
    AGENT = 2

class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

# TODO: actually use this
class Render(Enum):
    NOTHING = 0
    TRAINING = 1
    MOVES = 2

class GameStatus(Enum):
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

# Constants for rewards and penalties
# TODO: make these configurable form the outside
    reward_goal = 10.0
    penalty_move = -0.05
    penalty_already_visited = -0.1
    penalty_impossible_move = -0.5

    def __init__(self, maze, start_cell=(0, 0), goal_cell=None, max_steps=10000, render_mode=Render.NOTHING):
        self._original_maze = maze
        self.maze = np.copy(maze)
        self.start_cell = start_cell
        nrows, ncols = self.maze.shape
        self.goal_cell = goal_cell if goal_cell else (nrows - 1, ncols - 1)
        self.max_steps = max_steps
        self._validate_and_set_cells(nrows, ncols)

        self.render_mode = render_mode
        self.renderer = MazeRenderer(
            agent_color='dodgerblue',
            goal_color='gold',
            show_grid=False
        )
        self.reset(self.start_cell)

    def reset(self, start_cell):
        self.steps = 0
        self.agent_position = start_cell
        self.maze = np.copy(self._original_maze)
        row, col = start_cell
        self.maze[row][col] = CellType.AGENT
        self.total_reward_environment = 0.0
        self.visited = set()
        self.visited.add(start_cell)
        return self.agent_position
    
    def get_shape(self):
        return self.maze.shape

    def game_status(self):
        if self.agent_position == self.goal_cell:
            return GameStatus.SUCCESS
        if self.steps >= self.max_steps:
            return GameStatus.FAILURE
        else:
            return GameStatus.IN_PROGRESS

    def render(self):
        self.renderer.render(self, self.agent_position, reward_positions=[self.goal_cell])

    def step(self, action):
        step_reward = self._execute_and_reward(action)
        self.total_reward_environment += step_reward
        self.steps += 1
        game_status = self.game_status()
        new_agent_position = self.agent_position
        return new_agent_position, step_reward, game_status

    def _execute_and_reward(self, action):
        row, col = self.agent_position
        delta_row, delta_col = self.actions[action]
        new_row, new_col = row + delta_row, col + delta_col

        if not self._is_valid((new_row, new_col)):
            return self.penalty_impossible_move

        self.maze[self.agent_position] = CellType.EMPTY
        self.agent_position = (new_row, new_col)
        self.maze[self.agent_position] = CellType.AGENT

        if (new_row, new_col) in self.visited:
            return self.penalty_already_visited

        self.visited.add(self.agent_position)

        if self.agent_position == self.goal_cell:
            return self.reward_goal

        return self.penalty_move

    def _is_valid(self, position):
        r, c = position
        return 0 <= r < self.maze.shape[0] and 0 <= c < self.maze.shape[1] and self.maze[r][c] != CellType.WALL
    
    def _validate_and_set_cells(self, nrows, ncols):
        self.empty_cells = [(r, c) for r in range(nrows) for c in range(ncols) if self.maze[r][c] == CellType.EMPTY]
        if self.goal_cell in self.empty_cells:
            self.empty_cells.remove(self.goal_cell)

        if self.maze[self.goal_cell] == CellType.WALL:
            raise ValueError("Goal cell cannot be inside a wall.")

        if self.start_cell not in self.empty_cells:
            raise ValueError("The agent must start on an empty cell.")

        if self.start_cell == self.goal_cell:
            raise ValueError("The agent cannot start on the goal cell.")