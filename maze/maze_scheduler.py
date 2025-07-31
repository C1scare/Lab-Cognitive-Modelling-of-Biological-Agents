from maze.basic_maze import BasicMaze
from maze.maze_definitions import mazes
from typing import Tuple, List
import numpy as np

class MazeScheduler:
    """
    Schedules mazes and start positions for agent training and testing.

    - Cycles through a range of mazes (from `first` to `last`).
    - For each maze, cycles through all possible start positions in random order.
    - Provides methods to get the next start cell or load the next maze.

    Attributes:
        first: ID of the first maze in the sequence.
        last: ID of the last maze in the sequence.
        trials_start: Number of resets before switching to the next start position.
        trials_maze: Number of resets before switching to the next maze.
        random_seed: Seed for reproducible shuffling of start positions.
        reset_count: Counter for the number of resets.
        start_index: Index of the current start position.
        start_permutation: List of shuffled start positions for the current maze.
        maze: The current BasicMaze instance.
    """

    def __init__(
        self,
        first: int = 1,
        last: int = 25,
        trials_start: int = 1,
        trials_maze: int = 10,
        random_seed: int = 42
    ):
        """
        Initialize the MazeScheduler.

        Args:
            first: ID of the first maze to use.
            last: ID of the last maze to use.
            trials_start: Number of resets before switching to the next start position.
            trials_maze: Number of resets before switching to the next maze.
            random_seed: Seed for reproducible shuffling of start positions.
        """
        self.first = first
        self.current_maze_id = first
        self.last = last
        self.trials_start = trials_start
        self.trials_maze = trials_maze
        self.random_seed = random_seed
        self.reset_count = 0
        self.start_index = 0
        self.start_permutation = []
        self.maze: BasicMaze = self._load_maze(first)

    def next_start_cell(self) -> Tuple[int, int]:
        """
        Advances to the next start position if needed and returns it.
        Cycles after `trials_start` resets.

        Returns:
            The next start cell as a (row, col) tuple.
        """
        self.reset_count += 1

        # Cycle to next start position if needed
        if self.reset_count % self.trials_start == 0:
            self.start_index = (self.start_index + 1) % len(self.start_permutation)

        return self.start_permutation[self.start_index]

    def next_maze(self) -> None:
        """
        Loads the next maze in the sequence and resets start positions.
        """
        self.current_maze_id += 1
        if self.current_maze_id > self.last:
            self.current_maze_id = self.first
        self.maze = self._load_maze(self.current_maze_id)

    def _load_maze(self, maze_id: int) -> BasicMaze:
        """
        Loads a maze by ID and shuffles its start positions.

        Args:
            maze_id: The ID of the maze to load.

        Returns:
            A BasicMaze instance initialized with a shuffled start cell.

        Raises:
            ValueError: If the maze_id is out of range.
        """
        if maze_id < 1 or maze_id > 27:
            raise ValueError("Maze ID must be between 1 and 27.")
        maze_name = f"maze_{maze_id:02d}"
        maze_info = mazes[maze_name]
        maze_array = maze_info['maze']
        start_cells = maze_info['start_cells']
        # Shuffle start cells and convert to list of tuples
        rng = np.random.RandomState(self.random_seed)
        permuted = rng.permutation(len(start_cells))
        self.start_permutation: List[Tuple[int, int]] = [ (int(start_cells[i][0]), int(start_cells[i][1])) for i in permuted ]
        self.start_index = 0
        start_cell: Tuple[int, int] = self.start_permutation[self.start_index]
        goal_cell = maze_info['goal_cell']

        return BasicMaze(maze=maze_array, start_cell=start_cell, goal_cell=goal_cell, maze_ID=maze_id)