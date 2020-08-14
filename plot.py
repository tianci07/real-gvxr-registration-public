#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser();
parser.add_argument("--repeat", help="Repeat running for how many times", required=True);
parser.add_argument("--metrics", help="Choose single or all metrics, options=['ZNCC', 'SSIM', 'MI', 'GC', 'MAE', 'CS', 'SSD', 'GD', 'ALL']", required=True);
parser.add_argument("--downscale", help="Lower the image resolution using Guassian Pyramids, eg. 0 for no scaling, 1 for 1/4 (1/2 width and 1/2 height), 2 for 1/16, so on...")
args = parser.parse_args();
fitness = [];
with open('./results/fitness.txt') as file:
    for line in file:
        line = line.replace("FITNESS", "");
        fitness.append(float(line));

if (args.metrics == "ZNCC" or args.metrics == "SSIM" or args.metrics == "MI" or args.metrics =="GC"):
    fitness = [ -f for f in fitness];

plt.plot(np.arange(1, len(fitness)+1), fitness);
plt.xlabel('Generations');
plt.ylabel(args.metrics);
plt.savefig("./results/plot-"+args.metrics+"-run"+args.repeat+"-downscale"+args.downscale +".pdf");
