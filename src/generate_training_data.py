#BSD 3-Clause License
#
#Copyright (c) 2019, The Regents of the University of Minnesota
#
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#* Redistributions of source code must retain the above copyright notice, this
#  list of conditions and the following disclaimer.
#
#* Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#* Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 21:16:37 2019
This is the main code that runs simulated annleaing to generate the data
@author: Vidya A Chhabria
"""

import sys
import numpy as np
from create_template import define_templates
from T6_PSI_settings import T6_PSI_settings
from simulated_annealer import simulated_annealer
import random
from tqdm import tqdm
#from eliminate_templates import load_template_list

def main():
    # Read the json user input file and the current maps that need to be run
    # taken as an argument from the scripts
    settings_obj = T6_PSI_settings.load_obj()
    if len(sys.argv) == 3:
        map_start = int(sys.argv[1])
        num_maps = int(sys.argv[2])
        congestion_enabled = 1
    elif len(sys.argv) == 4:
        map_start = int(sys.argv[1])
        num_maps = int(sys.argv[2])
        if sys.argv[3] == "no_congestion":
            congestion_enabled = 0
        else:
            congestion_enabled = 1 
    else:
        map_start = 1
        num_maps = 15
        congestion_enabled = 1 
        print("Warning defaulting to %d %d and with congestion" % (map_start, num_maps))
        print(sys.argv)

    #print(num_maps)
    # Initialize the SA parameters
    T_init = 70
    T_final = 0.0005
    alpha_temp = 0.95
    num_moves_per_step = 5

    state = [] #np.zeros((num_maps, settings_obj.NUM_REGIONS))
    e = []#np.zeros(num_maps)
    max_drop = [] #np.zeros((num_maps, settings_obj.NUM_REGIONS))
    template_list = define_templates(settings_obj, generate_g=0)
    congestion = []
    all_templates = settings_obj.load_template_list()#range(settings_obj.NUM_TEMPLATES))
    size_region_x = int(settings_obj.WIDTH_REGION * 1e6)
    size_region_y = int(settings_obj.LENGTH_REGION * 1e6)
    current_maps = []
    for i in tqdm(range(num_maps)):
#        print(i)
        power_map_file = settings_obj.map_dir + "current_map_%d.csv" % (
            i + map_start)
        currents = np.genfromtxt(power_map_file, delimiter=',')
        for y in range(settings_obj.current_map_num_regions):
            for x in range(settings_obj.current_map_num_regions):
                print("%d %d "%(x,y))
                #region and neighbors
                current_region = np.zeros((3*size_region_x,3*size_region_y))
                init_state = np.zeros(9, int)
                if congestion_enabled == 1 :
                    signal_cong = [0.3 + 0.7*random.uniform(0, 1) for _ in range(9) ]
                else:
                    signal_cong = [0 for _ in range(9) ]
                if x == 0:
                    x_start = 0
                    x_end = x_start+2*size_region_x
                    if y == 0:
                        y_start = 0
                        y_end = y_start+2*size_region_y
                        current_region[size_region_x:,size_region_y:] = (
                            currents[x_start:x_end,y_start:y_end])
                    elif y == settings_obj.current_map_num_regions-1:
                        y_start = (y-1)*size_region_y
                        y_end = y_start+2*size_region_y
                        current_region[size_region_x:,0:2*size_region_y] = (
                            currents[x_start:x_end,y_start:y_end])
                    else:
                        y_start = (y-1)*size_region_y
                        y_end = y_start+3*size_region_y
                        current_region[size_region_x:,:] = (
                            currents[x_start:x_end,y_start:y_end])
                elif x == settings_obj.current_map_num_regions-1:
                    x_start = (x-1)*size_region_x
                    x_end = x_start+2*size_region_x
                    if y == 0:
                        y_start = 0
                        y_end = y_start+2*size_region_y
                        current_region[0:2*size_region_x,size_region_y:] = (
                            currents[x_start:x_end,y_start:y_end])
                    elif y == settings_obj.current_map_num_regions-1:
                        y_start = (y-1)*size_region_y
                        y_end = y_start+2*size_region_y
                        current_region[0:2*size_region_x,0:2*size_region_y] = (
                            currents[x_start:x_end,y_start:y_end])
                    else:
                        y_start = (y-1)*size_region_y
                        y_end = y_start+3*size_region_y
                        current_region[0:2*size_region_x,:] = (
                            currents[x_start:x_end,y_start:y_end])
                else:
                    x_start = (x-1)*size_region_x
                    x_end = x_start+3*size_region_x
                    if y == 0:
                        y_start = 0
                        y_end = y_start+2*size_region_y
                        current_region[:,size_region_y:] = (
                            currents[x_start:x_end,y_start:y_end])
                    elif y == settings_obj.current_map_num_regions-1:
                        y_start = (y-1)*size_region_y
                        y_end = y_start+2*size_region_y
                        current_region[:,0:2*size_region_y] = (
                            currents[x_start:x_end,y_start:y_end])
                    else:
                        y_start = (y-1)*size_region_y
                        y_end = y_start+3*size_region_y
                        current_region[:,:] = (
                            currents[x_start:x_end,y_start:y_end])
                    
                pdn_opt = simulated_annealer(init_state, T_init, T_final, 
                        alpha_temp, num_moves_per_step,
                        current_region,congestion_enabled)
                n_state, n_e, n_max_drop = pdn_opt.sim_anneal(
                    all_templates, template_list,signal_cong)
                state.append(n_state)
                max_drop.append(n_max_drop)
                congestion.append(signal_cong)
                current_maps.append(current_region.reshape(-1))
                e.append(n_e)
                #print(n_state,n_max_drop,signal_cong,n_e)
    with open(
            settings_obj.parallel_run_dir + 'max_drop_%d_to_%d.csv' %
        (map_start, map_start + num_maps - 1), 'w') as outfile:
        np.savetxt(outfile, max_drop, delimiter=',', fmt='%f')
    with open(
            settings_obj.parallel_run_dir + 'state_%d_to_%d.csv' %
        (map_start, map_start + num_maps - 1), 'w') as outfile:
        np.savetxt(outfile, state, delimiter=',', fmt='%d')
    with open(
            settings_obj.parallel_run_dir + 'energy_%d_to_%d.csv' %
        (map_start, map_start + num_maps - 1), 'w') as outfile:
        np.savetxt(outfile, e, delimiter=',', fmt='%f')
    if congestion_enabled ==1:
        with open(
                settings_obj.parallel_run_dir + 'congest_%d_to_%d.csv' %
            (map_start, map_start + num_maps - 1), 'w') as outfile:
            np.savetxt(outfile,congestion, delimiter=',', fmt='%f')
    with open(
            settings_obj.parallel_run_dir + 'current_maps_%d_to_%d.csv' %
        (map_start, map_start + num_maps - 1), 'w') as outfile:
        np.savetxt(outfile,current_maps, delimiter=',', fmt='%f')


if __name__ == '__main__':
    main()
