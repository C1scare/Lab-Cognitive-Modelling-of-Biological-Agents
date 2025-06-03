from enum import Enum, IntEnum
from typing import Tuple, Optional, Dict, Set, List
import numpy as np
import numpy.typing as npt
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

class GameStatus(Enum):
    SUCCESS = 0
    FAILURE = 1
    IN_PROGRESS = 2

class BasicMaze:
    """
    A simple grid-based maze environment for agent navigation and reinforcement learning.
    The agent moves within a fixed-size maze with walls, empty cells, and a goal.
    """

    actions: Dict[Action, Tuple[int, int]] = {
        Action.UP: (-1, 0),
        Action.DOWN: (1, 0),
        Action.LEFT: (0, -1),
        Action.RIGHT: (0, 1)
    }

    # Reward configuration
    

    def __init__(
        self, 
        maze: npt.NDArray[np.int_],
        start_cell: Tuple[int, int] = (0, 0),
        goal_cell: Optional[Tuple[int, int]] = None,
        max_steps: int = 10000,
        reward_goal: float = 10.0,
        penalty_move: float = -0.05,
        penalty_already_visited: float = -0.1,
        penalty_impossible_move: float = -0.5
        ) -> None:
        """
        Initialize the maze environment.

        Args:
            maze: A 2D numpy array representing the maze layout.
            start_cell: Starting coordinates of the agent.
            goal_cell: Goal coordinates (defaults to bottom-right).
            max_steps: Maximum allowed steps before failure.
            reward_goal: Reward for reaching the goal.
            penalty_move: Penalty for each move.
            penalty_already_visited: Penalty for moving to an already visited cell.
            penalty_impossible_move: Penalty for attempting an impossible move.
        """
        self._original_maze: npt.NDArray[np.int_] = maze
        self.maze: npt.NDArray[np.int_] = np.copy(maze)
        self.start_cell: Tuple[int, int] = start_cell
        nrows, ncols = self.maze.shape
        self.goal_cell: Tuple[int, int] = goal_cell if goal_cell else (nrows - 1, ncols - 1)
        self.max_steps: int = max_steps
        self.reward_goal: float = reward_goal
        self.penalty_move: float = penalty_move
        self.penalty_already_visited: float = penalty_already_visited
        self.penalty_impossible_move: float = penalty_impossible_move

        self._validate_and_set_cells(nrows, ncols)

        self.renderer: MazeRenderer = MazeRenderer(
            agent_color='dodgerblue',
            goal_color='gold',
            show_grid=False
        )
        self.reset(self.start_cell)

    def reset(self, start_cell: Tuple[int, int]) -> Tuple[int, int]:
        """
        Reset the maze to its initial state and place the agent.

        Args:
            start_cell: The new starting position for the agent.

        Returns:
            The initial agent position.
        """
        self.steps: int = 0
        self.agent_position: Tuple[int, int] = start_cell
        self.maze = np.copy(self._original_maze)
        row, col = start_cell
        self.maze[row][col] = CellType.AGENT
        self.total_reward_environment: float = 0.0
        self.visited: Set[Tuple[int, int]] = {start_cell}
        return self.agent_position

    def get_shape(self) -> Tuple[int, int]:
        """Return the shape of the maze."""
        shape = self.maze.shape
        return (int(shape[0]), int(shape[1]))

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
        """Render the maze using the MazeRenderer."""
        #self.renderer.render(self, self.agent_position, reward_positions=[self.goal_cell])
        self.renderer.render_maze(self.maze, agent_position=self.agent_position, reward_positions=[self.goal_cell], visited=self.visited)

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

    def get_action_space(self) -> List[Action]:
        """
        Get the available actions in the maze.

        Returns:
            A list of possible actions.
        """
        return list(self.actions.keys())

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
            (r, c)
            for r in range(nrows)
            for c in range(ncols)
            if self.maze[r][c] == CellType.EMPTY
        ]
        if self.goal_cell in self.empty_cells:
            self.empty_cells.remove(self.goal_cell)

        if self.maze[self.goal_cell] == CellType.WALL:
            raise ValueError("Goal cell cannot be inside a wall.")

        if self.start_cell not in self.empty_cells:
            raise ValueError("The agent must start on an empty cell.")

        if self.start_cell == self.goal_cell:
            raise ValueError("The agent cannot start on the goal cell.")
        
        sr, sc = self.start_cell
        start_neighbors: List[Tuple[int, int]] = [(sr - 1, sc), (sr + 1, sc), (sr, sc - 1), (sr, sc + 1)]
        valid_start_neighbor_exists: bool = any(
            0 <= r < nrows and 0 <= c < ncols and self.maze[r][c] != CellType.WALL
            for r, c in start_neighbors
        )
        if not valid_start_neighbor_exists:
            raise ValueError("Start cell must have at least one accessible neighboring cell.")
        
        gr, gc = self.goal_cell
        goal_neighbors: List[Tuple[int, int]] = [(gr - 1, gc), (gr + 1, gc), (gr, gc - 1), (gr, gc + 1)]
        valid_goal_neighbor_exists: bool = any(
            0 <= r < nrows and 0 <= c < ncols and self.maze[r][c] != CellType.WALL
            for r, c in goal_neighbors
        )
        if not valid_goal_neighbor_exists:
            raise ValueError("Goal cell must have at least one accessible neighboring cell.")

