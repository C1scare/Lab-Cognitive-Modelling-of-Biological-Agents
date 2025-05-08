import numpy as np
from maze.basic_maze import BasicMaze, Action

# Define a simple 5x5 maze
maze_array = np.array([
    [0, 0, 1, 0, 0],
    [0, 1, 1, 0, 1],
    [0, 0, 0, 0, 0],
    [1, 0, 1, 1, 0],
    [0, 0, 0, 0, 0]
])

# Initialize the maze environment
env = BasicMaze(maze_array, start_cell=(0, 0), goal_cell=(4, 4))

# Interact with the environment
env.render()
_, reward, status = env.step(Action.DOWN)
env.render()
_, reward, status = env.step(Action.DOWN)
env.render()
