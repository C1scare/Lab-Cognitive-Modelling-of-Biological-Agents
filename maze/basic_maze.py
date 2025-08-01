from enum import Enum, IntEnum
from typing import Tuple, Optional, Dict, Set, List
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from maze.maze_renderer import MazeRenderer
from maze.maze_definitions import mazes

class CellType(IntEnum):
    EMPTY = 0
    WALL = 1
    AGENT = 2

class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class GameStatus(Enum):
    SUCCESS = 0
    FAILURE = 1
    IN_PROGRESS = 2

class BasicMaze:
    """
    A simple grid-based maze environment for agent navigation and reinforcement learning.

    The agent moves within a fixed-size maze with walls, empty cells, and a goal.
    Supports agent movement, reward calculation, step limits, and rendering.

    Attributes:
        maze: The current maze layout as a 2D numpy array.
        start_cell: The starting position of the agent.
        goal_cell: The goal position in the maze.
        max_steps: Maximum allowed steps before failure.
        agent_position: The agent's current position.
        steps: Number of steps taken in the current episode.
        total_reward_environment: Cumulative reward collected in the current episode.
        visited: Set of visited cells.
        empty_cells: List of empty (non-wall) cells in the maze.
        renderer: MazeRenderer instance for visualization.
        maze_ID: Optional identifier for the maze.
        random_seed: Random seed for reproducibility.
    """

    actions: Dict[Action, Tuple[int, int]] = {
        Action.UP: (-1, 0),
        Action.DOWN: (1, 0),
        Action.LEFT: (0, -1),
        Action.RIGHT: (0, 1)
    }

    # Reward configuration
    reward_goal: float = 10.0
    penalty_move: float = -1
    penalty_already_visited: float = -1 #unused in our experiments
    penalty_impossible_move: float = -5

    def __init__(
        self, 
        maze: npt.NDArray[np.int_],
        start_cell: Tuple[int, int] = (0, 0),
        goal_cell: Optional[Tuple[int, int]] = None,
        max_steps: int = 30,
        random_seed: int = 42,
        maze_ID: Optional[int] = None
    ) -> None:
        """
        Initialize the maze environment.

        Args:
            maze: A 2D numpy array representing the maze layout.
            start_cell: Starting coordinates of the agent.
            goal_cell: Goal coordinates (defaults to bottom-right).
            max_steps: Maximum allowed steps before failure.
            random_seed: Random seed for reproducibility.
            maze_ID: Optional identifier for the maze.
        """
        self._original_maze: npt.NDArray[np.int_] = maze
        self.maze: npt.NDArray[np.int_] = np.copy(maze)
        self.start_cell: Tuple[int, int] = start_cell
        nrows, ncols = self.maze.shape
        self.goal_cell: Tuple[int, int] = goal_cell if goal_cell else (nrows - 1, ncols - 1)
        self.max_steps: int = max_steps
        self._validate_and_set_cells(nrows, ncols)

        self.renderer: MazeRenderer = MazeRenderer(
            agent_color='dodgerblue',
            goal_color='gold',
            show_grid=False
        )
        self.reset(self.start_cell)

        self.random_seed: int = random_seed
        self.maze_ID: Optional[int] = maze_ID

    def reset(self, start_cell: Tuple[int, int]) -> Tuple[int, int]:
        """
        Reset the maze to its initial state and place the agent.

        Args:
            start_cell: The new starting position for the agent.

        Returns:
            The initial agent position.
        """
        self.steps: int = 0
        self.start_cell = start_cell
        self.agent_position: Tuple[int, int] = self.start_cell
        self.maze = np.copy(self._original_maze)
        row, col = self.start_cell
        self.maze[row][col] = CellType.AGENT
        self.total_reward_environment: float = 0.0
        self.visited: Set[Tuple[int, int]] = {self.start_cell}

        return self.agent_position

    def get_shape(self) -> Tuple[int, int]:
        """
        Return the shape of the maze as a tuple (rows, columns).

        Returns:
            Tuple of (number of rows, number of columns).
        """
        maze_shape: Tuple[int, int] = (int(self.maze.shape[0]), int(self.maze.shape[1]))
        return maze_shape
    
    def game_status(self) -> GameStatus:
        """
        Check the current game status.

        Returns:
            GameStatus.SUCCESS if goal reached,
            GameStatus.FAILURE if max steps exceeded,
            GameStatus.IN_PROGRESS otherwise.
        """
        if self.agent_position == self.goal_cell:
            return GameStatus.SUCCESS
        if self.steps >= self.max_steps:
            return GameStatus.FAILURE
        return GameStatus.IN_PROGRESS

    def render(self) -> None:
        """
        Render the maze using the MazeRenderer.
        """
        self.renderer.render(self, self.agent_position, reward_positions=[self.goal_cell])

    def step(self, action: Action) -> Tuple[Tuple[int, int], float, GameStatus]:
        """
        Perform a step in the maze given an action.

        Args:
            action: The movement direction.

        Returns:
            A tuple of (new agent position, reward received, current game status).
        """
        step_reward: float = self._execute_and_reward(action)
        self.total_reward_environment += step_reward
        self.steps += 1
        game_status: GameStatus = self.game_status()
        new_agent_position: Tuple[int, int] = self.agent_position
        return new_agent_position, step_reward, game_status

    def _execute_and_reward(self, action: Action) -> float:
        """
        Move the agent and return the resulting reward.

        Args:
            action: The action to apply.

        Returns:
            The reward from taking the action.
        """
        row, col = self.agent_position
        delta_row, delta_col = self.actions[action]
        new_row, new_col = row + delta_row, col + delta_col

        if not self._is_valid((new_row, new_col)):
            return self.penalty_impossible_move

        self.maze[self.agent_position] = CellType.EMPTY
        self.agent_position = (new_row, new_col)
        self.maze[self.agent_position] = CellType.AGENT

        if self.agent_position in self.visited:
            return self.penalty_already_visited

        self.visited.add(self.agent_position)

        if self.agent_position == self.goal_cell:
            return self.reward_goal

        return self.penalty_move

    def _is_valid(self, position: Tuple[int, int]) -> bool:
        """
        Check if a position is a valid, non-wall cell within bounds.

        Args:
            position: Cell to check.

        Returns:
            True if position is valid; False otherwise.
        """
        r, c = position
        return 0 <= r < self.maze.shape[0] and 0 <= c < self.maze.shape[1] and self.maze[r][c] != CellType.WALL

    def _validate_and_set_cells(self, nrows: int, ncols: int) -> None:
        """
        Validate that the start and goal cells are on valid empty spaces.

        Raises:
            ValueError: If start or goal are invalid positions.
        """
        self.empty_cells: list[Tuple[int, int]] = [
            (int(r), int(c))
            for r in range(nrows)
            for c in range(ncols)
            if self.maze[r][c] == CellType.EMPTY
        ]
        sr, sc = map(int, self.start_cell)
        self.start_cell = (sr, sc)
        gr, gc = map(int, self.goal_cell)
        self.goal_cell = (gr, gc)

        if self.goal_cell in self.empty_cells:
            self.empty_cells.remove(self.goal_cell)

        if self.maze[self.goal_cell] == CellType.WALL:
            raise ValueError("Goal cell cannot be inside a wall.")

        if self.start_cell not in self.empty_cells:
            raise ValueError("The agent must start on an empty cell.")

        if self.start_cell == self.goal_cell:
            raise ValueError("The agent cannot start on the goal cell.")
        
        start_neighbors: List[Tuple[int, int]] = [(sr - 1, sc), (sr + 1, sc), (sr, sc - 1), (sr, sc + 1)]
        valid_start_neighbor_exists: bool = any(
            0 <= r < nrows and 0 <= c < ncols and self.maze[r][c] != CellType.WALL
            for r, c in start_neighbors
        )
        if not valid_start_neighbor_exists:
            raise ValueError("Start cell must have at least one accessible neighboring cell.")
        
        goal_neighbors: List[Tuple[int, int]] = [(gr - 1, gc), (gr + 1, gc), (gr, gc - 1), (gr, gc + 1)]
        valid_goal_neighbor_exists: bool = any(
            0 <= r < nrows and 0 <= c < ncols and self.maze[r][c] != CellType.WALL
            for r, c in goal_neighbors
        )
        if not valid_goal_neighbor_exists:
            raise ValueError("Goal cell must have at least one accessible neighboring cell.")

    @staticmethod
    def render_maze_figure(maze_ID: int, start_cell_id:int = 0) -> None:
        """
        Render a maze from maze_definitions.py.

        Args:
            maze_ID: Unique identifier for the maze.
            start_cell_id: Index of the start cell to use from the maze's start cells.

        Raises:
            ValueError: If maze_ID or start_cell_id are out of valid range.
        """
        if maze_ID < 1 or maze_ID > 27:
            raise ValueError("Maze ID must be between 1 and 27.")
        if start_cell_id < 0 or start_cell_id >= len(mazes[f"maze_{maze_ID:02d}"]['start_cells']):
            raise ValueError(f"Invalid start cell ID {start_cell_id} for maze {maze_ID}.")
        maze_id = maze_ID
        maze_name = f"maze_{maze_id:02d}"
        maze_info = mazes[maze_name]
        maze_array = maze_info['maze']
        start_cells = maze_info['start_cells']
        start_cell = start_cells[start_cell_id]
        goal_cell = maze_info['goal_cell']

        maze = BasicMaze(maze=maze_array, start_cell=start_cell, goal_cell=goal_cell, maze_ID=maze_id)
        maze.render()
        plt.show()
        
        