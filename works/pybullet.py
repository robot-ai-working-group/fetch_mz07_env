import os
import pybullet_data
import numpy as np
from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
import time
import pybullet as p
import random
import matplotlib.pyplot as plt

rootdir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import sys
sys.path.append(rootdir)
from envs.pybullet_mz07_env import Mz07GymEnv
from algorithms.q_learning import *

def encode_pos(joints):
    obs=joints[0]+joints[1]*91+joints[2]*91*31
    return obs

if __name__=="__main__":

    q_table = np.zeros((91*31*31, 2*2*2))
    ql=q_learning(q_table)
    env = Mz07GymEnv()

    fig, ax = plt.subplots(1,1)
    len_list=[]
    lines, = ax.plot([0],[0])

    for episode in range(1,501):

        env.reset()

        joints=[0,0,0]
        step=0
        jointPoses=[0 for i in range(12)]
        done=False
        total=0
        gripperState = p.getLinkState(env._robot.robotUid, env._robot.endEffectorIndex)
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
            # print(p.getJointState(env._robot.kukaUid,6))

            gripperState = p.getLinkState(env._robot.robotUid, env._robot.endEffectorIndex)
            pos = gripperState[0]
            len=np.linalg.norm(np.array(pos)-np.array(gripperPos))
            gripperPos=pos
            total+=len

            if (joints[0]==90 and joints[1]==30 and joints[2]==30):
                done=True

            reward=-10
            if done:
                reward=-total

            # print("step: ",step, joints, done, len)
            last_obs=encode_pos(last_joints)
            obs=encode_pos(joints)
            print("action:{},last_obs:{}, obs:{}".format(action,last_obs,obs),end="\r")
            ql.update_q_table(action, last_obs, obs, reward)

        print("done:", episode, "len:", total)

        len_list.append(total)
        x=[i for i in range(1,episode+1)]
        lines.set_data(x,len_list)
        ax.set_xlim((1,episode))
        ax.set_ylim((0,1.5))
        plt.pause(.01)

    print(len_list)
