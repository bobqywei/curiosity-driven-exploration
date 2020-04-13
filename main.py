import logging
import yaml
import os
import argparse
import torch
import shutil

from a2c import A2C
from tensorboardX import SummaryWriter

from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack

parser = argparse.ArgumentParser(description="Curiosity-driven A2C")
parser.add_argument('--config', default='configs/main.yaml')

def get_logger(config):
    # get a logger for output logging info on screen and save in dir
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # create file handler which save logger in dir
    # save everything include DEBUG info
    file_handler = logging.FileHandler(config['outdir'] + '/logging.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # only log to console if tqdm is off
    if not config['use_tqdm']:
        # create console handler which print logger on screen
        # only print higher level info if necessary
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

def main():
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    writer = None
    # Will ERROR if outdir already exists
    if not os.path.exists(config['outdir']):
        os.makedirs(config['outdir'])
        if config['use_tensorboard']:
            os.makedirs(os.path.join(config['outdir'], 'tensorboard'))
            writer = SummaryWriter(os.path.join(config['outdir'], 'tensorboard'))
        # save a copy of the config file
        shutil.copyfile(args.config, os.path.join(config['outdir'], 'config.yaml'))
    else:
        print("ERROR: directory \'./{}\' already exists!".format(config['outdir']))
        raise EnvironmentError

    logger = get_logger(config)

    env = make_atari_env(config['task'], num_env=config['parallel_envs'], seed=config['seed'])
    env = VecFrameStack(env, n_stack=config['state_frames'])

    device = torch.device('cuda') if config['use_gpu'] else torch.device('cpu')

    a2c = A2C(config, env, device, logger, writer)
    a2c.train()


if __name__ == '__main__':
    main()