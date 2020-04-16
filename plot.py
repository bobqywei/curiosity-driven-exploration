import matplotlib.pyplot as plt
import argparse
import os
import csv
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--name')
parser.add_argument('--paths', nargs='+')
args = parser.parse_args()

# exponential moving average
def smooth(scalars, weight):
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point 
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


col_names = ['Step', 'Value']
data = []
for path in args.paths:
    df = pd.read_csv(path, names=col_names, skipinitialspace=True)
    data.append((df.Step.values[1:].astype(int), df.Value.values[1:].astype(float)))

colours = ['r', 'y', 'c']
labels = ['A2C', 'A2C + Entropy', 'A2C + Entropy + Curiosity']
i = 0
for d in data:
    plt.plot(d[0], d[1], alpha=0.2, color=colours[i])
    plt.plot(d[0], smooth(d[1], 0.95), color=colours[i], label=labels[i])
    i = (i+1) % 3
plt.legend()
plt.xlabel('Training Update Steps')
plt.ylabel('Extrinsic Reward per Episode')
plt.title(args.name)
plt.show()