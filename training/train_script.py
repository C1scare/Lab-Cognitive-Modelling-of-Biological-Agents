from maze.basic_maze import GameStatus

def train_agent(env, agent, episodes=500):
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

        if episode % 50 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Status: {status.name}")
