import numpy as np
import gym
import gym.envs.toy_text.frozen_lake as fl
import matplotlib.pyplot as plt
import time

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from algorithms.q_learning import *


env = fl.FrozenLakeEnv(is_slippery=False)

if __name__=="__main__":

    # initialize q-_q_table
    q_table = np.zeros((16, 4))
    ql=q_learning(q_table)
    steps=[]

    for episode in range(100):
        obs = env.reset()
        done = False
        step = 0

        while not done:

            current_obs=obs

            action=ql.get_action(current_obs)
            obs, reward, done, info = env.step(action)

            # overwirte reward
            if done:
                reward = 100 if reward == 1 else -100
            else:
                reward = -1

            ql.update_q_table(action, current_obs, obs, reward)

            env.render()
            step+=1

        steps.append(step if reward>0 else 0)

    # show diagram
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(100), steps)
    ax.set_xlabel('episode')
    ax.set_ylabel('step count')
    plt.legend(loc='best')
    plt.show()
