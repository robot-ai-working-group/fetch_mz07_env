import os
import pybullet_data
import numpy as np

rootdir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import sys
sys.path.append(rootdir)
from envs.pybullet_mz07_env import Mz07GymEnv

if __name__=="__main__":
    env = Mz07GymEnv(
        urdfRoot=pybullet_data.getDataPath(),
        renders=True,
        isDiscrete=False,
        maxSteps=10000000)

    while True:
       env.step(env.action_space.sample())
       # print("step")
