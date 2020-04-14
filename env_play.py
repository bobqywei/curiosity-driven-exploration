import gym
import random
import numpy as np

env = gym.make('SkiingNoFrameskip-v4')
obs = env.reset()
n = env.action_space.n
for i in range(10000):
    env.render()
    action = random.randint(0,n-1)
    obs, rewards, dones, infos = env.step(action)
    print(rewards)
if 'episode' in infos.keys():
    print(infos['episode']['r'])

if dones > 0:
    print("hello")
    obs = env.reset()
else:
    print("render")