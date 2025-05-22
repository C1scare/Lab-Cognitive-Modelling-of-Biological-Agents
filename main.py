import numpy as np
import numpy.typing as npt
from maze.basic_maze import BasicMaze, GameStatus
from agents.q_learning import QLearningAgent
from training.train_script import train_agent, plot_rewards
import matplotlib.pyplot as plt
from maze.maze_definitions import mazes
from typing import Any, Dict, List, Tuple

if __name__ == "__main__":

    maze_info: Dict[str, Any] = mazes["maze_15"]  # change this to the desired maze name
    maze_array: npt.NDArray[np.int_] = maze_info["maze"]
    start_cells: List[Tuple[int, int]] = maze_info["start_cells"]
    start_cell: Tuple[int, int] = start_cells[8]  # change this to the desired start cell
    goal_cell: Tuple[int, int] = maze_info["goal_cell"]

    env = BasicMaze(maze=maze_array, start_cell=start_cell, goal_cell=goal_cell)
    agent = QLearningAgent(maze_shape=env.get_shape(), action_Space=env.get_action_space())

    rewards = train_agent(env, agent, episodes=1000)
    plot_rewards(rewards)
    
    # Test run
    state = env.reset(env.start_cell)

    plt.ion()  # Turn on interactive mode for live updating
    fig, ax = plt.subplots()
    env.render()
    plt.pause(0.3)

    while True:
        action = agent.choose_action(state)
        state, _, status = env.step(action)
        env.render()
        plt.pause(0.1)  # Wait briefly to animate

        if status != GameStatus.IN_PROGRESS:
            print("Test run ended with status:", status)
            break

    plt.ioff() # type: ignore
    plt.show()
