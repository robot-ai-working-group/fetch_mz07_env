import os
import pybullet_data
import numpy as np
from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
import time
import pybullet as p
import random

rootdir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import sys
sys.path.append(rootdir)
from envs.pybullet_mz07_env import Mz07GymEnv
from algorithms.q_learning import *

if __name__=="__main__":

    q_table = np.zeros((91*31*31, 2*2*2))
    ql=q_learning(q_table)
    env = Mz07GymEnv()

    for episode in range(100):

        env.reset()

        joints=[0,0,0]
        step=0
        jointPoses=[0 for i in range(12)]
        done=False
        total=0
        gripperState = p.getLinkState(env._kuka.kukaUid, env._kuka.kukaGripperIndex)
        gripperPos = gripperState[0]

        while not done:
            step+=1

            action=random.randint(0,7)

            last_joints=joints

            if joints[0]<90 and action & 1:
                joints[0]+=1
            if joints[1]<30 and action & 2:
                joints[1]+=1
            if joints[2]<30 and action & 4:
                joints[2]+=1

            jointPoses[0]=joints[0]/180*3.14
            jointPoses[1]=joints[1]/180*3.14
            jointPoses[2]=joints[2]/180*3.14
            env.step(jointPoses)
            # print(p.getJointState(env._kuka.kukaUid,6))

            gripperState = p.getLinkState(env._kuka.kukaUid, 6)
            pos = gripperState[0]
            len=np.linalg.norm(np.array(pos)-np.array(gripperPos))
            gripperPos=pos
            total+=len

            if (joints[0]==90 and joints[1]==30 and joints[2]==30):
                done=True

            reward=-100
            if done:
                reward=-total

            # print("step: ",step, joints, done, len)
            last_obs=last_joints[0]+last_joints[1]*90+last_joints[2]*90*30
            obs=joints[0]+joints[1]*90+joints[2]*90*30
            ql.update_q_table(action, last_obs, obs, reward)

        print("done:", episode, "len:", total)
        time.sleep(10)
