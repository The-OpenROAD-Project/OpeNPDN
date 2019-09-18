#!/usr/bin/env python3

import sys
import subprocess
import os
import numpy as np

print("Creating a tar ball of all training data")
os.system('tar -cvzf - input/current_maps/current_map_*.csv --remove-files | split -b 40M - "input/current_maps/current_maps.tgz_part"')

#os.system('zip -s 40m -m -q input/current_maps/current_maps.zip input/current_maps/current_map_*.csv')

print("Creating a tar ball of all training data")
os.system('tar -cvzf - output/CNN_*.csv --remove-files | split -b 40M - "output/CNN_data.tgz_part"')

#os.system('zip -s 40m -m -q output/CNN_data.zip output/CNN_*.csv')

print("Creating a tar ball of all checkpoints")
#os.system('zip -s 40m -m -q output/checkpoints/checkpoints.zip output/checkpoints/*')
os.system('tar -cvzf - output/checkpoints/* --remove-files | split -b 40M - "output/checkpoints/checkpoints.tgz_part"')
