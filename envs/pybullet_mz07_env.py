import os
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
import random
import pybullet_data
from pkg_resources import parse_version

rootdir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import sys
sys.path.append(rootdir)
from envs.pybullet_mz07 import Mz07

largeValObservation = 100

RENDER_HEIGHT = 720
RENDER_WIDTH = 960


class Mz07GymEnv():

    def __init__(self):
        self._isDiscrete = False
        self._timeStep = 1. / 240.
        self._urdfRoot = pybullet_data.getDataPath()
        self._actionRepeat = 1
        self._isEnableSelfCollision = False
        self._observation = []
        self._envStepCounter = 0
        self._renders = True
        self._maxSteps = 1000
        self.terminated = 0
        self._cam_dist = 1.3
        self._cam_yaw = 180
        self._cam_pitch = -40

        self._p = p
        if self._renders:
          cid = p.connect(p.SHARED_MEMORY)
          if (cid < 0):
            cid = p.connect(p.GUI)
          p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
        else:
          p.connect(p.DIRECT)

        self._robot = None
        self.seed()
        self.reset()

        observationDim = len(self.getExtendedObservation())

        observation_high = np.array([largeValObservation] * observationDim)
        if (self._isDiscrete):
          self.action_space = spaces.Discrete(7)
        else:
          action_dim = 3
          self._action_bound = 1
          action_high = np.array([self._action_bound] * action_dim)
          self.action_space = spaces.Box(-action_high, action_high)
        self.observation_space = spaces.Box(-observation_high, observation_high)
        self.viewer = None

        self._timeStep = 1. / 1000.
        self._actionRepeat = 1

    def reset(self):

        self.terminated = 0
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -10)
        # if self._robot==None:
        self._robot = Mz07(urdfRootPath=os.path.join(rootdir,"envs/assets"), timeStep=self._timeStep)
        self._envStepCounter = 0
        p.stepSimulation()
        self._observation = self.getExtendedObservation()
        return np.array(self._observation)

    def __del__(self):
        p.disconnect()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def getExtendedObservation(self):
        self._observation = self._robot.getObservation()
        return self._observation

    def step(self, action):

        # jointPoses=[0 for i in range(12)]
        for i in range(self._actionRepeat):
            self._robot.applyAction(action)
            p.stepSimulation()
            if self._termination():
                break
            self._envStepCounter += 1
        if self._renders:
            time.sleep(self._timeStep)
        self._observation = self.getExtendedObservation()

        done = self._termination()
        actionCost=0
        reward = self._reward() - actionCost
        return np.array(self._observation), reward, done, {}

    def render(self, mode="rgb_array", close=False):
        if mode != "rgb_array":
            return np.array([])

        base_pos, orn = self._p.getBasePositionAndOrientation(self._robot.robotUid)
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self._p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
            nearVal=0.1,
            farVal=100.0)
        (_, _, px, _, _) = self._p.getCameraImage(
            width=RENDER_WIDTH,
            height=RENDER_HEIGHT,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=self._p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _termination(self):
        return False

    def _reward(self):
        return -100
