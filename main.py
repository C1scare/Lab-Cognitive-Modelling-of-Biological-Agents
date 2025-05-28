import numpy as np
from maze.basic_maze import BasicMaze, GameStatus
from agents.q_learning import QLearningAgent
from agents.bayesian_q_learning import BayesianQLearningAgent
from training.train_script import train_agent, plot_rewards
import matplotlib.pyplot as plt
from agents.agent_type import AgentType
from agents.noisy_agent import NoisyAgent
from agents.curious_agent import CuriousAgent
from agents.noise_mode import NoiseMode
from agents.hyperparameter import Hyperparameter

if __name__ == "__main__":
    maze_array = np.array([
        [0, 0, 1, 0],
        [1, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 1, 1, 0]
    ])

    env = BasicMaze(maze=maze_array, start_cell=(0, 0), goal_cell=(3, 3))

    agent_type = AgentType.CURIOUS_AGENT  # Change to AgentType.Q_LEARNING for Bayesian Q-learning
    agent = None

    if agent_type == AgentType.Q_LEARNING:
        print("Using Q-learning agent")
        agent = QLearningAgent(
            maze_shape=env.get_shape(),
            action_Space=env.actions
        )
    
    elif agent_type == AgentType.BAYESIAN_Q_LEARNING:
        print("Using Bayesian Q-learning agent")
        agent = BayesianQLearningAgent(
            maze_shape=env.get_shape(),
            action_space=env.actions,
            hyperparameters=Hyperparameter(
                alpha=0.1,
                gamma=0.99,
                epsilon=0.2,
                mu_init=0.0,
                sigma_sq_init=2.0,
                obs_noise_variance=0.1
            )
        )
    elif agent_type == AgentType.NOISY_AGENT:
        noise_mode = NoiseMode.PERCEPTUAL  # Change to NoiseMode.NEURAL or NoiseMode.BOTH as needed
        print(f"Using Noisy agent with {noise_mode.value}")
        agent = NoisyAgent(
            maze_shape=env.get_shape(),
            action_space=env.actions,
            hyperparameters=Hyperparameter(
                alpha=0.1,
                gamma=0.99,
                epsilon=0.2,
                mu_init=0.0,
                sigma_sq_init=2.0,
                obs_noise_variance=0.1,
                k_pn=1,
                sigma_nn=1,
                noise_mode=noise_mode
            )
        )
    elif agent_type == AgentType.CURIOUS_AGENT:
        print("Using Curious agent")
        agent = CuriousAgent(
            maze_shape=env.get_shape(),
            action_space=env.actions,
            hyperparameters=Hyperparameter(
                gamma=0.99, 
                epsilon=0.2, 
                mu_init=0.0, 
                sigma_sq_init=2.0, 
                obs_noise_variance=0.1,
                curiosity_init=0.5,
                alpha_C=0.1,
                surprise_weight=1.0,
                novelty_weight=1.0,
                usefulness_weight=1.0,
                uncertainty_weight=1.0,
                beta_T=0.1,
                beta_U=0.1,
                beta_N=0.1
            )
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
