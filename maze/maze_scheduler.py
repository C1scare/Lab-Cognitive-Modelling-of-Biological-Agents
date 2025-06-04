from maze.basic_maze import BasicMaze


class MazeScheduler:
    def __init__(self, trials: int = 10, random_seed: int = 42):
        # Load first maze
        self.maze:BasicMaze = None # Placeholder for maze object, has to be loaded here from file
        self.trials = trials
        self.random_seed = random_seed


    def next_maze(self) -> BasicMaze:
        # TODO: get current maze ID
        # TODO: get next maze with ID + 1 % 25
        pass



    