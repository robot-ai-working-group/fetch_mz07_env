import os
from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
import pybullet as p
import random
import numpy as np
import time
import pybullet_data
from gym import spaces

rootdir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import sys
sys.path.append(rootdir)
from envs.pybullet_mz07 import Mz07

class Mz07GymEnv(KukaGymEnv):

    def __init__(self):
        super().__init__(
            urdfRoot=pybullet_data.getDataPath(),
            renders=True,
            isDiscrete=False,
            maxSteps=10000000)

        self._timeStep = 1. / 10.
        self._actionRepeat = 1

    def reset(self):
        #print("KukaGymEnv _reset")
        self.terminated = 0
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -1])

        p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5000000, 0.00000, -.820000,
                   0.000000, 0.000000, 0.0, 1.0)

        xpos = 0.55 + 0.12 * random.random()
        ypos = 0 + 0.2 * random.random()
        ang = 3.14 * 0.5 + 3.1415925438 * random.random()
        orn = p.getQuaternionFromEuler([0, 0, ang])
        self.blockUid = p.loadURDF(os.path.join(self._urdfRoot, "block.urdf"), xpos, ypos, -0.15,
                                   orn[0], orn[1], orn[2], orn[3])

        p.setGravity(0, 0, -10)
        self._kuka = Mz07(urdfRootPath=os.path.join(rootdir,"envs/assets"), timeStep=self._timeStep)
        self._envStepCounter = 0
        p.stepSimulation()
        self._observation = self.getExtendedObservation()
        return np.array(self._observation)

    def step(self, action):

        # jointPoses=[0 for i in range(12)]
        for i in range(self._actionRepeat):
            self._kuka.applyAction(action)
            p.stepSimulation()
            if self._termination():
                break
            self._envStepCounter += 1
        if self._renders:
            time.sleep(self._timeStep)
        self._observation = self.getExtendedObservation()

        done = self._termination()
        # npaction = np.array([
        #     action[3]
        # ])
        # actionCost = np.linalg.norm(npaction) * 10.
        actionCost=0
        reward = self._reward() - actionCost
        return np.array(self._observation), reward, done, {}
