import yaml
import argparse
import gym
import torch
import random
import a2c
from a2c import ActorCritic
from a2c import A2C
import numpy as np
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack
from tqdm import tqdm 
from stable_baselines.common.vec_env import VecVideoRecorder, DummyVecEnv

video_folder = 'videos/'
video_length = 10

parser = argparse.ArgumentParser(description="Curiosity-driven A2C")
parser.add_argument('--config', default='configs/main.yaml')

args = parser.parse_args()

with open(args.config) as f:
    config = yaml.safe_load(f)

env = make_atari_env(config['task'], num_env=config['parallel_envs'], seed=config['seed'])
env = VecFrameStack(env, n_stack=config['state_frames'])
env = VecVideoRecorder(env, video_folder,
                       record_video_trigger=lambda x: x == 0, video_length=video_length,
                       name_prefix="random-agent-{}".format(config['task']))
obs = env.reset()
n = env.action_space.n

model = ActorCritic(n, config)
device = torch.device('cuda') if config['use_gpu'] else torch.device('cpu')
model.load_state_dict(torch.load('checkpoints\pong_noEnt\model_recent_ckpt', map_location=device))
model.eval()

for i in tqdm(range(video_length+1)):
    #env.render(mode='rgb_array')
    tensor = torch.from_numpy(obs.astype(np.float32).transpose((0, 3, 1, 2))) / 255
    tensor = torch.nn.functional.interpolate(tensor, scale_factor=48/tensor.shape[-1])
    action, _, _, _ = model.forward(tensor.to(device))
    obs, _, dones, _ = env.step(action)
    if dones.sum() > 0:
        obs = env.reset()
env.close()