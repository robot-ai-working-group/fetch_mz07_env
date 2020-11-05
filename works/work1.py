import numpy as np
import gym

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import envs.reach_mz07_env as reach_env

#import gym.envs.robotics.fetch.reach as env_reach

env = reach_env.ReachMz07Env()
#env = gym.make('FetchReach-v1')

def policy(observation, desired_goal):
    # Here you would implement your smarter policy. In this case,
    # we just sample random actions.
    return env.action_space.sample()

for _ in range(1000):
    obs = env.reset()
    done = False

    while not done:
        env.render()
        action = policy(obs['observation'], obs['desired_goal'])
        obs, reward, done, info = env.step(action)

        # If we want, we can substitute a goal here and re-compute
        # the reward. For instance, we can just pretend that the desired
        # goal was what we achieved all along.
        substitute_goal = obs['achieved_goal'].copy()
        substitute_reward = env.compute_reward(
            obs['achieved_goal'], substitute_goal, info)
        print('reward is {}, substitute_reward is {}'.format(
            reward, substitute_reward))
