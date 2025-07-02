from maze.basic_maze import BasicMaze
from maze.maze_definitions import mazes
from typing import Tuple
import numpy as np

class MazeScheduler:
    """
    Handles scheduling of mazes and start positions for agent training and testing.

    - Cycles through a range of mazes (from `first` to `last`).
    - For each maze, cycles through all possible start positions in random order.
    - Provides methods to get the next start cell or load the next maze.
    """

    def __init__(
        self,
        first: int = 1,
        last: int = 25,
        trials_start: int = 1,
        trials_maze: int = 10,
        random_seed: int = 42
    ):
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
        """
        self.reset_count += 1

        # Cycle to next start position if needed
        if self.reset_count % self.trials_start == 0:
            self.start_index = (self.start_index + 1) % len(self.start_permutation)

        return self.start_permutation[self.start_index]

    def next_maze(self) -> BasicMaze:
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
        self.start_permutation = [tuple(map(int, start_cells[i])) for i in permuted]
        self.start_index = 0
        start_cell: Tuple[int, int] = self.start_permutation[self.start_index]
        goal_cell = maze_info['goal_cell']

        return BasicMaze(maze=maze_array, start_cell=start_cell, goal_cell=goal_cell, maze_ID=maze_id)