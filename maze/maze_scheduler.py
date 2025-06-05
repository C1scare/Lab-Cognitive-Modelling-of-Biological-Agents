from maze.basic_maze import BasicMaze
import numpy as np


class MazeScheduler:
    def __init__(self, trials: int = 10, random_seed: int = 42):
        # Load first maze
        #self.maze:BasicMaze = None # Placeholder for maze object, has to be loaded here from file
        self.trials = trials
        self.random_seed = random_seed
        maze_outline = np.array([
            [0, 0, 1, 0],
            [1, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 1, 1, 0]
        ])
        self.maze:BasicMaze = BasicMaze(maze=maze_outline, start_cell=(0, 0), goal_cell=(3, 3))


    def next_maze(self) -> BasicMaze:
        # TODO: get current maze ID
        # TODO: get next maze with ID + 1 % 25
        return self.maze



    