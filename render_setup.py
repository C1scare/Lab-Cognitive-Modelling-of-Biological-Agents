import matplotlib.pyplot as plt
from maze.maze_definitions import mazes
from maze.maze_renderer import MazeRenderer

maze_name = "maze_01"  # Change this to the desired maze name
maze_info = mazes[maze_name]
maze_array = maze_info["maze"]
goal = maze_info["goal_cell"]
starts = maze_info["start_cells"]

renderer = MazeRenderer()
plt.figure(figsize=(6, 6))
renderer.render_setup(maze=maze_array, start_positions=starts, goal_position=goal)
plt.title(f"{maze_name} Setup: Start Positions and Goal")
plt.show()
