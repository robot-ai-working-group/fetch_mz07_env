import os
import gym
import pybullet_envs
from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
import pybullet_envs.bullet.kuka as kuka
import pybullet_data
import pybullet as p
import random
import numpy as np

rootdir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CustomKukaGymEnv(KukaGymEnv):

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
    self._kuka = kuka.Kuka(urdfRootPath=os.path.join(rootdir,"envs/assets"), timeStep=self._timeStep)
    self._envStepCounter = 0
    p.stepSimulation()
    self._observation = self.getExtendedObservation()
    return np.array(self._observation)


if __name__=="__main__":
    env = CustomKukaGymEnv(
        urdfRoot=pybullet_data.getDataPath(),
        renders=True,
        isDiscrete=False,
        maxSteps=10000000)

    while True:
       env.step(env.action_space.sample())
       # print("step")