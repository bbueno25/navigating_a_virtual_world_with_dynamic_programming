import agent
import environment
import numpy as np

env = environment.make()
n_actions = env.action_space.n
n_state_features = env.observation_space.shape[0]
dqn_agent = agent.DQNAgent(n_state_features, n_actions)

if __name__ == '__main__':
    for episode in range(environment.EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, n_state_features])
        for t in range(environment.EPISODE_LENGTH):
            # Predict next action using NN Value Function Approximation
            action = dqn_agent.get_action(state)
            # Interact with the environment and observe new state and reward
            next_state, reward, terminated, info = env.step(action)
            # Huge negative reward if failed
            if terminated:
                reward = -100
            # Remember agent's experience: state / action / reward / next state
            next_state = np.reshape(next_state, [1, n_state_features])
            dqn_agent.remember(state, action, reward, next_state, terminated)
            # Change the current state
            state = next_state
            # Print statistics if agent failed and quit inner loop
            if terminated:
                print('Episode: {} of {}'.format(episode, environment.EPISODES))
                print('Score: {} seconds'.format(t))
                print('Exploration rate: {:.4}'.format(dqn_agent.epsilon))
                break
        # Re-train Value Function Approximation model if we have enough examples in memory
        if len(dqn_agent.memory) >= environment.BATCH_SIZE:
            dqn_agent.experience_replay(environment.BATCH_SIZE)
        # Save trained agent every once in a while
        if episode % 100 == 0:
            dqn_agent.save('models/{}.h5'.format(environment.NAME))
