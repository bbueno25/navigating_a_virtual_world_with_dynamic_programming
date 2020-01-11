import agent
import environment
import numpy as np

if __name__ == '__main__':
    env = environment.Environment().make_env()
    n_actions = env.action_space.n
    n_state_features = env.observation_space.shape[0]
    # Initialize DQN agent
    dqn_agent = agent.DQNAgent(n_state_features, n_actions, epsilon=0.0)
    # Load pre-trained agent
    dqn_agent.load('.\\models\\{}.h5'.format(env.id))
    for episode in range(environment.EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, n_state_features])
        for t in range(environment.EPISODE_LENGTH):
            # Visualize environment
            env.render()
            # Predict next action using NN Value Function Approximation
            action = dqn_agent.get_action(state)
            # Interact with the environment and observe new state and reward
            next_state, reward, terminated, info = env.step(action)
            next_state = np.reshape(next_state, [1, n_state_features])
            # Change the current state
            state = next_state
            # Print statistics if agent failed and quit inner loop
            if terminated:
                print('Episode: {} of {}'.format(episode, environment.EPISODES))
                print('score: {}s, exploration rate: {:.4}'.format(t, dqn_agent.epsilon))
                break
