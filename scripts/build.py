#!/usr/bin/env python3

import sys
import subprocess
import os
import numpy as np
import glob

#TODO check if already unziped? 

#os.system('cat input/current_maps/current_maps.* > input/current_maps/current_maps_merged.zip')
#os.system('unzip -q input/current_maps/current_maps_merged.zip ')
print("Unpacking the current_maps")
os.system('cat input/current_maps/current_maps.tgz_part* | tar -xzvf - && rm input/current_maps/current_maps.tgz_part*')
#os.system('cat output/CNN_data.z* > output/CNN_data_merged.zip')
#os.system('unzip -q output/CNN_data_merged.zip')
print("Unpacking the training data")
if glob.glob('output/CNN_data.tgz_part*'):
    os.system('cat output/CNN_data.tgz_part* | tar -xzvf - && rm output/CNN_data.tgz_part*')
else:
    print("CNN data with congestion not found, proceeding")
if glob.glob('output/CNN_data_wo_cong.tgz_part*'):
    os.system('cat output/CNN_data_wo_cong.tgz_part* | tar -xzvf - && rm output/CNN_data_wo_cong.tgz_part*')
else:
    print("CNN data without congestion not found, proceeding")
#os.system('cat output/checkpoints/checkpoints.z* > output/checkpoints/checkpoints_merged.zip')
#os.system('unzip -q output/checkpoints/checkpoints_merged.zip')
#os.remove('input/current_maps/current_maps_merged.zip')    
#os.remove('output/CNN_data_merged.zip')
#os.remove('output/checkpoints/checkpoints_merged.zip')
#os.system('rm output/CNN_data.z*')
#os.system('rm output/checkpoints/checkpoints.z*')
#os.system('rm input/current_maps/current_maps.z*')
print("Unpacking the checkpoints with and without congestion")
if not os.path.exists('output/checkpoints/'):
    os.system('mkdir -p output/checkpoints/') 
    print("Checkpoints with congetion undefined, proceeding")
else:
    if glob.glob('output/checkpoints/checkpoints.tgz_part*'):
        os.system('cat output/checkpoints/checkpoints.tgz_part* | tar -xzvf - && rm output/checkpoints/checkpoints.tgz_part*')
    else:
        print("Checkpoints with congetion undefined, proceeding")
if not os.path.exists('output/checkpoints_wo_cong/'):
    os.system('mkdir -p output/checkpoints_wo_cong/ ')
    print("Checkpoints without congetion undefined, proceeding")
else:
    if glob.glob('output/checkpoints_wo_cong/checkpoints.tgz_part*'):
        os.system('cat output/checkpoints_wo_cong/checkpoints.tgz_part* | tar -xzvf - && rm output/checkpoints_wo_cong/checkpoints.tgz_part*')
    else:
        print("Checkpoints without congetion undefined, proceeding")
os.system('mkdir -p work templates work/parallel_runs')

