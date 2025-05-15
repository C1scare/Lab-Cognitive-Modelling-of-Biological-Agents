from maze.basic_maze import GameStatus

def train_agent(
    env, 
    agent, 
    episodes: int = 500,
    decay_epsilon: bool = True,
) -> None:
    """
    Train a reinforcement learning agent in a given environment.

    Args:
        env: The environment with `reset()` and `step()` methods.
        agent: The agent with `choose_action()`, `learn()`, and optionally `decay_epsilon()` methods.
        episodes: Number of training episodes to run.
        decay_epsilon: Whether to decay the agent's exploration rate after each episode.
    """
    for episode in range(episodes):
        state = env.reset(env.start_cell)
        total_reward = 0

        while True:
            action = agent.choose_action(state)
            next_state, reward, status = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward

            if status != GameStatus.IN_PROGRESS:
                break

        if decay_epsilon:
            agent.decay_epsilon()

        if episode % 50 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Status: {status.name}")
