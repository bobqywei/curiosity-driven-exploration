import logging
import yaml
import os
import argparse
import torch
from a2c import A2C
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description="Curiosity-driven A2C")
parser.add_argument('--config', default='configs/main.yaml')

def main():
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda') if config['use_gpu'] else torch.device('cpu')

    a2c = A2C(config, env, device)
    A2C.train()