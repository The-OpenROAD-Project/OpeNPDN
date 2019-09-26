#!/usr/bin/env python3

import sys
import subprocess
import os
import numpy as np
import glob

print("Creating a tar ball of all raw training data")
os.system('tar -cvzf - input/current_maps/current_map_*.csv --remove-files | split -b 40M - "input/current_maps/current_maps.tgz_part"')

#os.system('zip -s 40m -m -q input/current_maps/current_maps.zip input/current_maps/current_map_*.csv')

# wiout congestion must be processed frist for the wildcards to work
print("Creating a tar ball of all training data")
if glob.glob('output/CNN_wo_cong_*'):
    os.system('tar -cvzf - output/CNN_wo_cong_*.csv --remove-files | split -b 40M - "output/CNN_data_wo_cong.tgz_part"')
else:
    print("CNN_data without congestion not found, proceeding")
if glob.glob('output/CNN_*'):
    os.system('tar -cvzf - output/CNN_*.csv --remove-files | split -b 40M - "output/CNN_data.tgz_part"')
else:
    print("CNN_data with congestion not found, proceeding")

#os.system('zip -s 40m -m -q output/CNN_data.zip output/CNN_*.csv')

print("Creating a tar ball of all checkpoints with and without congestion")
#os.system('zip -s 40m -m -q output/checkpoints/checkpoints.zip output/checkpoints/*')
if glob.glob('output/checkpoints/*'):
    os.system('tar -cvzf - output/checkpoints/* --remove-files | split -b 40M - "output/checkpoints/checkpoints.tgz_part"')
else:
    print("Checkpoints with congestion not found, doing nothing")
if glob.glob('output/checkpoints_wo_cong/*'):
    os.system('tar -cvzf - output/checkpoints_wo_cong/* --remove-files | split -b 40M - "output/checkpoints_wo_cong/checkpoints.tgz_part"')
else:
    print("Checkpoints without congestion not found, doing nothing")
