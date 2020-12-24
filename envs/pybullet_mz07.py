import os
import pybullet as p
import numpy as np
import math

class Mz07():

    def __init__(self, urdfRootPath, timeStep=0.01):
        self.urdfRootPath = urdfRootPath
        self.timeStep = timeStep
        self.maxVelocity = 35
        self.maxForce = 200.
        self.useInverseKinematics = 1
        self.useSimulation = 1
        self.useNullSpace = 21
        self.useOrientation = 1
        self.endEffectorIndex = 5
        self.reset()

    def reset(self):

        objects = p.loadSDF(os.path.join(self.urdfRootPath, "mz07_with_gripper.sdf"))
        self.robotUid = objects[0]
        p.resetBasePositionAndOrientation(
            self.robotUid,
            [-0.100000, 0.000000, 0.070000],
            [0.000000, 0.000000, 0.000000, 1.000000])
        self.jointPositions = [0, 0, 0, 0, 0, 0]
        self.numJoints = p.getNumJoints(self.robotUid)

        for jointIndex in range(self.numJoints):
            p.resetJointState(self.robotUid, jointIndex, self.jointPositions[jointIndex])
            p.setJointMotorControl2(
                self.robotUid,
                jointIndex,
                p.POSITION_CONTROL,
                targetPosition=self.jointPositions[jointIndex],
                force=self.maxForce)

        self.motorNames = []
        self.motorIndices = []
        for i in range(self.numJoints):
            jointInfo = p.getJointInfo(self.robotUid, i)
            qIndex = jointInfo[3]
            if qIndex > -1:
                self.motorNames.append(str(jointInfo[1]))
                self.motorIndices.append(i)

    def getObservation(self):
        observation = []
        state = p.getLinkState(self.robotUid, self.endEffectorIndex)
        pos = state[0]
        orn = state[1]
        euler = p.getEulerFromQuaternion(orn)

        observation.extend(list(pos))
        observation.extend(list(euler))

        return observation

    def applyAction(self, jointPoses):

        if (self.useSimulation):
            for i in range(self.endEffectorIndex):
                p.setJointMotorControl2(
                    self.robotUid,
                    i,
                    p.POSITION_CONTROL,
                    targetPosition=jointPoses[i],
                    force=self.maxForce)
