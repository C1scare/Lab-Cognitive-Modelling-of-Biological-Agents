from maze.basic_maze import GameStatus
from typing import List, Tuple
import matplotlib.pyplot as plt
from maze.basic_maze import BasicMaze, Action, GameStatus
from agents.base_agent import BaseAgent

def train_agent(
    env: BasicMaze, 
    agent: BaseAgent, 
    episodes: int = 500,
    decay_epsilon: bool = True,
) -> List[float]:
    """
    Train a reinforcement learning agent in a given environment.

    Args:
        env: The environment with `reset()` and `step()` methods.
        agent: The agent with `choose_action()`, `learn()`, and optionally `decay_epsilon()` methods.
        episodes: Number of training episodes to run.
        decay_epsilon: Whether to decay the agent's exploration rate after each episode.
    
    Returns:
        List of total rewards received in each episode.
    
    """
    episode_rewards: List[float] = []

    for episode in range(episodes):
        state = env.reset(env.start_cell)
        total_reward = 0

        while True:
            action: Action = agent.choose_action(state)
            next_state: Tuple[int, int]
            reward: float
            status: GameStatus 
            next_state, reward, status = env.step(action)
            agent.learn(state, action, reward, next_state)
            state: Tuple[int, int] = next_state
            total_reward += reward

            if status != GameStatus.IN_PROGRESS:
                break
            
        # Decay epsilon after each episode if applicable
        if decay_epsilon:
            agent.decay_epsilon()

        episode_rewards.append(total_reward)

        if episode % 50 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Status: {status.name}")

    return episode_rewards

def plot_rewards(rewards: List[float]) -> None:
    """
    Plot the total rewards received in each episode.

    Args:
        rewards: List of total rewards received in each episode.
    """

    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards Over Episodes')
    plt.grid(True)
    plt.show()  
