from maze.basic_maze import GameStatus
from typing import List, Tuple
from maze.basic_maze import BasicMaze, Action, GameStatus
from maze.maze_scheduler import MazeScheduler
from agents.base_agent import BaseAgent
from agents.curious_agent import CuriousAgent
from agents.bayesian_q_learning import BayesianQLearningAgent
import numpy as np
from pydantic import BaseModel, Field


class ExperimentResult(BaseModel):
    cumulative_reward: List[float] = Field(alias="cumulative_reward") 
    success_rate: float = Field(alias="success_rate")
    average_reward: float = Field(alias="average_reward")
    max_reward: float = Field(alias="max_reward")
    learning_speed: float = Field(alias="learning_speed")
    best_path_length: int = Field(alias="best_path_length")
    trajectory_history: dict[int, List[Tuple[Tuple[int, int], Tuple[int, int]]]] = Field(alias="trajectory_history", default_factory=dict)
    maze_history: dict[int, Tuple[BasicMaze, Tuple[int,int]]] = Field(alias="maze_history", default_factory=list)
    curiosity_history: dict[int, dict[Tuple[Tuple[int, int], Tuple[int, int]], float]] = Field(alias="curiosity_history", default_factory=dict)
    uncertainty_history: dict[int, dict[Tuple[Tuple[int, int], Tuple[int, int]], float]] = Field(alias="uncertainty_history", default_factory=dict)
    q_mean_history: dict[int, dict[Tuple[Tuple[int, int], Tuple[int, int]], float]] = Field(alias="q_mean_history", default_factory=dict)
    uncertainties:List[float] = Field(alias="uncertainties", default_factory=list)
    curiosity: list[float] = Field(alias="curiosity", default_factory=list)

    class Config:
        arbitrary_types_allowed = True


def train_agent(
    maze_scheduler: MazeScheduler, 
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
    env:BasicMaze = maze_scheduler.maze
    episode_rewards: List[float] = []
    uncertainties: List[float] = []
    curiosity = [] if isinstance(agent, CuriousAgent) else list(np.zeros(episodes))
    success_count = 0
    trajectory_history: dict[int, List[Tuple[Tuple[int, int],Tuple[int, int]]]] = {}
    maze_history: dict[int, Tuple[BasicMaze, Tuple[int,int]]] = {}
    curiosity_history = {} if isinstance(agent, CuriousAgent) else {}
    uncertainty_history = {} if isinstance(agent, BayesianQLearningAgent) else {}
    q_mean_history = {} if isinstance(agent, BayesianQLearningAgent) else {}


    for episode in range(episodes):
        state = env.reset(env.start_cell)
        maze_history[episode] = (env, env.start_cell)
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

            trajectory_history[episode].append((state, next_state))
            if isinstance(agent, CuriousAgent):
                curiosity_map = agent.transform_curiosity_map(env)
                curiosity_history[episode] = curiosity_map
            
            if isinstance(agent, BayesianQLearningAgent):
                uncertainty_map = agent.transform_q_dist_map(env, variance=True)
                uncertainty_history[episode] = uncertainty_map
                q_mean_map = agent.transform_q_dist_map(env)
                q_mean_history[episode] = q_mean_map

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
        uncertainties.append(np.sum(agent.q_dist_table[:, :, :, 1], axis=(0, 1, 2))) if isinstance(agent, BayesianQLearningAgent) else None
        curiosity.append(np.sum(agent.curiosity, axis=(0, 1, 2))) if isinstance(agent, CuriousAgent) else None


        if episode % 50 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Status: {status.name}")
        
        if episode % maze_scheduler.trials == 0 and episode > 0:
            maze_scheduler.next_maze()
            env = maze_scheduler.maze


    return ExperimentResult(
        cumulative_reward=episode_rewards,
        success_rate=success_count / episodes,
        average_reward=sum(episode_rewards) / episodes,
        max_reward=max(episode_rewards),
        learning_speed=-1 * np.argmax(np.array(episode_rewards)),
        best_path_length=len(trajectory_history[np.argmax(np.array(episode_rewards))]),
        trajectory_history=trajectory_history,
        maze_history=maze_history,
        curiosity_history=curiosity_history,
        uncertainties=uncertainties,
        uncertainty_history=uncertainty_history,
        q_mean_history=q_mean_history,
        curiosity=curiosity
    )
                            
