import numpy as np
import gym

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import envs.reach_mz07_env as reach_env

env = reach_env.ReachMz07Env()

def moveto(action, env):
    last_pos=np.array([10,0,0])
    for _ in range(1000):
        obs, reward, done, info = env.step(action)
        achieved_goal=obs['achieved_goal']
        dist=np.linalg.norm(achieved_goal-last_pos)
        if dist<0.0001:
            break
        last_pos=achieved_goal
        env.render()

    return (obs, reward, done, info)


# pos_ctrl, gripper_ctrl = action[:3], action[3]
def policy(observation, desired_goal, achieved_goal, step):

    actions=[
        [0.5,0.5,0.5,0],
        [-0.5,0.5,0.5,0],
        [-0.5,-0.5,0.5,0],
        [0.5,-0.5,0.5,0]
    ]
    # action=env.action_space.sample()

    return np.array(actions[step%4])

if __name__=="__main__":
    for episode in range(100):
        obs = env.reset()
        done = False
        step = 0
        while not done:
            step+=1
            action = policy(obs['observation'], obs['desired_goal'], obs['achieved_goal'], step)

            print(action)
            obs, reward, done, info = moveto(action, env)


            # If we want, we can substitute a goal here and re-compute
            # the reward. For instance, we can just pretend that the desired
            # goal was what we achieved all along.
            substitute_goal = obs['achieved_goal'].copy()
            substitute_reward = env.compute_reward(
                obs['achieved_goal'], substitute_goal, info)
            print('reward is {}, substitute_reward is {}'.format(
                reward, substitute_reward))
