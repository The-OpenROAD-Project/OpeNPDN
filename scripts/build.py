#!/usr/bin/env python3

import sys
import subprocess
import os
import numpy as np

#TODO check if already unziped? 

#os.system('cat input/current_maps/current_maps.* > input/current_maps/current_maps_merged.zip')
#os.system('unzip -q input/current_maps/current_maps_merged.zip ')
#os.system('cat output/CNN_data.z* > output/CNN_data_merged.zip')
#os.system('unzip -q output/CNN_data_merged.zip')
#os.system('cat output/checkpoints/checkpoints.z* > output/checkpoints/checkpoints_merged.zip')
#os.system('unzip -q output/checkpoints/checkpoints_merged.zip')
#os.remove('input/current_maps/current_maps_merged.zip')    
#os.remove('output/CNN_data_merged.zip')
#os.remove('output/checkpoints/checkpoints_merged.zip')
#os.system('rm output/CNN_data.z*')
#os.system('rm output/checkpoints/checkpoints.z*')
#os.system('rm input/current_maps/current_maps.z*')
print("Unpacking the checkpoints")
os.system('cat output/checkpoints/checkpoints.tgz_part* | tar -xzvf - && rm output/checkpoints/checkpoints.tgz_part*')
