import numpy as np
from maze.basic_maze import BasicMaze, GameStatus
from agents.q_learning import QLearningAgent
from training.train_script import train_agent
import matplotlib.pyplot as plt

if __name__ == "__main__":
    maze_array = np.array([
        [0, 0, 1, 0],
        [1, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 1, 1, 0]
    ])

    env = BasicMaze(maze=maze_array, start_cell=(0, 0), goal_cell=(3, 3))
    agent = QLearningAgent(maze_shape=env.get_shape(), action_Space=env.actions)

    train_agent(env, agent, episodes=1000)

    # Test run
state = env.reset(env.start_cell)

plt.ion()  # Turn on interactive mode for live updating
fig, ax = plt.subplots()
env.render()
plt.pause(0.5)

while True:
    action = agent.choose_action(state)
    state, _, status = env.step(action)
    env.render()
    plt.pause(0.5)  # Wait briefly to animate

    if status != GameStatus.IN_PROGRESS:
        print("Test run ended with status:", status)
        break

plt.ioff()
plt.show()
