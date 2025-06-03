from maze.basic_maze import GameStatus
from typing import List, Tuple
from maze.basic_maze import BasicMaze, Action, GameStatus
from agents.base_agent import BaseAgent
import numpy as np
from pydantic import BaseModel, Field


class ExperimentResult(BaseModel):
    cumulative_reward: List[float] = Field(alias="cumulative_reward") 
    success_rate: float = Field(alias="success_rate")
    average_reward: float = Field(alias="average_reward")
    max_reward: float = Field(alias="max_reward")
    learning_speed: float = Field(alias="learning_speed")
    best_path_length: int = Field(alias="best_path_length")
    trajectory_history: dict[int, List[Tuple[int, int]]] = Field(alias="trajectory_history", default_factory=dict)


def train_agent(
    env: BasicMaze, 
    agent: BaseAgent, 
    episodes: int = 500,
    decay_epsilon: bool = True
) -> ExperimentResult:
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
    success_count = 0
    trajectory_history: dict[int, List[Tuple[int, int]]] = {}

    for episode in range(episodes):
        state = env.reset(env.start_cell)
        total_reward = 0
        if episode not in trajectory_history:
            trajectory_history[episode] = []

        while True:
            action: Action = agent.choose_action(state)
            next_state: Tuple[int, int]
            reward: float
            status: GameStatus 
            next_state, reward, status = env.step(action)
            done = status != GameStatus.IN_PROGRESS
            agent.learn(state, action, reward, next_state, done)

            trajectory_history[episode].append(next_state)

            state: Tuple[int, int] = next_state
            total_reward += reward

            if done:
                if status == GameStatus.SUCCESS:
                    success_count += 1
                break

        if hasattr(agent, "decay_epsilon"):
            if decay_epsilon:
                agent.decay_epsilon()

        episode_rewards.append(total_reward)

        if episode % 50 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Status: {status.name}")


    return ExperimentResult(
        cumulative_reward=episode_rewards,
        success_rate=success_count / episodes,
        average_reward=sum(episode_rewards) / episodes,
        max_reward=max(episode_rewards),
        learning_speed=-1 * np.argmax(np.array(episode_rewards)),
        best_path_length=len(trajectory_history[np.argmax(np.array(episode_rewards))]),
        trajectory_history=trajectory_history
    )
                            
