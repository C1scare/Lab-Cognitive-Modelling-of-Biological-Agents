from maze.basic_maze import BasicMaze


class MazeScheduler:
    def __init__(self, trials: int = 10):
        # Load first maze
        self.maze:BasicMaze = None # Placeholder for maze object, has to be loaded here from file
        self.trials = trials


    def next_maze(self) -> BasicMaze:
        # TODO: get current maze ID
        # TODO: get next maze with ID + 1 % 25
        pass



    