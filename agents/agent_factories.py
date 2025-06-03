from agents.q_learning import QLearningAgent
from experiment.hyperparameters import Hyperparameters
from maze.basic_maze import BasicMaze

def q_learning_factory(env: BasicMaze, hp: Hyperparameters) -> QLearningAgent:
    return QLearningAgent(
        maze_shape=env.get_shape(),
        action_Space=env.get_action_space(),
        alpha=hp.alpha,
        gamma=hp.gamma,
        epsilon=hp.epsilon
    )
