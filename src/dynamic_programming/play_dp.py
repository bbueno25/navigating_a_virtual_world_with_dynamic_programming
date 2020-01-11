import dp
import environment as gym_env
import numpy as np

def play_episodes(environment, n_episodes, policy):
    wins = 0
    total_reward = 0
    for _ in range(n_episodes):
        terminated = False
        state = environment.reset()
        while not terminated:
            # Select best action to perform in a current state
            action = np.argmax(policy[state])
            # Perform an action an observe how environment acted in response
            next_state, reward, terminated, _ = environment.step(action)
            # Summarize total reward
            total_reward += reward
            # Update current state
            state = next_state
            # Calculate number of wins over episodes
            if terminated and reward == 1.0:
                wins += 1
    average_reward = total_reward / n_episodes
    return wins, total_reward, average_reward

if __name__ == '__main__':
    # Action mappings
    action_mapping = {0: 'UP', 1: 'RIGHT', 2: 'DOWN', 3: 'LEFT'}
    # Functions to find best policy
    solvers = [('Policy Iteration', dp.policy_iteration),
               ('Value Iteration', dp.value_iteration)]
    for iteration_name, iteration_func in solvers:
        # Load a Frozen Lake environment
        environment = gym_env.make('FrozenLake_bbueno5000-v0')
        # Search for an optimal policy using policy iteration
        policy, V = iteration_func(environment.env)
        print('\nFinal policy derived using {}:'.format(iteration_name))
        print(' '.join([action_mapping[action] for action in np.argmax(policy, axis=1)]))
        # Apply best policy to the real environment
        wins, total_reward, average_reward = play_episodes(environment, gym_env.EPISODES, policy)
        print('{}:number of wins {}:episodes {}'.format(iteration_name, gym_env.EPISODES, wins))
        print('{}:average reward {}:episodes {} \n'.format(iteration_name, gym_env.EPISODES, average_reward))
