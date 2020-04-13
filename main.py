import logging
import yaml
import os
import argparse
import torch
from a2c import A2C
from tensorboardX import SummaryWriter

from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack

parser = argparse.ArgumentParser(description="Curiosity-driven A2C")
parser.add_argument('--config', default='configs/main.yaml')

def main():
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    env = make_atari_env(config['task'], num_env=config['parallel_envs'], seed=1234)
    env = VecFrameStack(env, n_stack=config['state_frames'])

    device = torch.device('cuda') if config['use_gpu'] else torch.device('cpu')

    a2c = A2C(config, env, device)
    a2c.train()


if __name__ == '__main__':
    main()