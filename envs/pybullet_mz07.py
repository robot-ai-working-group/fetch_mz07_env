import os
import pybullet_envs.bullet.kuka as kuka
import pybullet as p

class Mz07(kuka.Kuka):

    def reset(self):
        objects = p.loadSDF(os.path.join(self.urdfRootPath, "mz07_with_gripper.sdf"))
        self.kukaUid = objects[0]
        #for i in range (p.getNumJoints(self.kukaUid)):
        #  print(p.getJointInfo(self.kukaUid,i))
        p.resetBasePositionAndOrientation(self.kukaUid, [-0.100000, 0.000000, 0.070000],
                                          [0.000000, 0.000000, 0.000000, 1.000000])
        self.jointPositions = [
            0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684, -0.006539, 0.000048,
            -0.299912, 0.000000, -0.000043, 0.299960, 0.000000, -0.000200
        ]
        self.numJoints = p.getNumJoints(self.kukaUid)
        for jointIndex in range(self.numJoints):
          p.resetJointState(self.kukaUid, jointIndex, self.jointPositions[jointIndex])
          p.setJointMotorControl2(self.kukaUid,
                                  jointIndex,
                                  p.POSITION_CONTROL,
                                  targetPosition=self.jointPositions[jointIndex],
                                  force=self.maxForce)

        self.trayUid = p.loadURDF(os.path.join(self.urdfRootPath, "tray/tray.urdf"), 0.640000,
                                  0.075000, -0.190000, 0.000000, 0.000000, 1.000000, 0.000000)
        self.endEffectorPos = [0.537, 0.0, 0.5]
        self.endEffectorAngle = 0

        self.motorNames = []
        self.motorIndices = []

        for i in range(self.numJoints):
          jointInfo = p.getJointInfo(self.kukaUid, i)
          qIndex = jointInfo[3]
          if qIndex > -1:
            #print("motorname")
            #print(jointInfo[1])
            self.motorNames.append(str(jointInfo[1]))
            self.motorIndices.append(i)
