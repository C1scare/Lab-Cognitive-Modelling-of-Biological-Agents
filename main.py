import numpy as np
from maze.basic_maze import BasicMaze, GameStatus
from agents.q_learning import QLearningAgent
from agents.bayesian_q_learning import BayesianQLearningAgent
from training.train_script import train_agent, plot_rewards
import matplotlib.pyplot as plt
from agents.agent_type import AgentType

if __name__ == "__main__":
    maze_array = np.array([
        [0, 0, 1, 0],
        [1, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 1, 1, 0]
    ])

    env = BasicMaze(maze=maze_array, start_cell=(0, 0), goal_cell=(3, 3))

    AgentType = AgentType.BAYESIAN_Q_LEARNING  # Change to AgentType.Q_LEARNING for Bayesian Q-learning
    agent = None

    if AgentType.Q_LEARNING == AgentType.Q_LEARNING:
        agent = QLearningAgent(maze_shape=env.get_shape(), action_Space=env.actions)
    
    elif AgentType.Q_LEARNING == AgentType.BAYESIAN_Q_LEARNING:
        agent = BayesianQLearningAgent(
            maze_shape=env.get_shape(),
            action_space=env.actions,
            alpha=0.1,
            gamma=0.99,
            epsilon=0.2,
            mu_init=0.0,
            sigma_sq_init=2.0,
            obs_noise_variance=0.1
        )
    else:
        raise ValueError("Unsupported agent type specified.")

    rewards = train_agent(env, agent, episodes=1000)
    plot_rewards(rewards)
    
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
