import gym
import gym_frozen_lake

EPISODES = 10000
EPISODE_LENGTH = 1000

def make(env_id):
    environment = gym.make(env_id)
    environment.max_episode_steps = EPISODE_LENGTH
    return environment
