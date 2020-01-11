import gym

BATCH_SIZE = 128
EPISODES = 1500
EPISODE_LENGTH = 1000
NAME = 'CartPole-v1'

def make():
    environment = gym.make(NAME)
    environment.max_episode_steps = EPISODE_LENGTH
    return environment
