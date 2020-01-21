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
This is the main code that runs simulated annleaing to decide the templates
@author:Vidya A Chhabria
"""

import sys
import numpy as np
from create_template_new import load_templates
from create_template_new import template
from create_template_new import node
from T6_PSI_settings import T6_PSI_settings
from construct_eqn_new import construct_eqn
from scipy import sparse as sparse_mat
import matplotlib.image as img
import math
import time
import re
from pprint import pprint



current_map_file = "./work/current_map_processed.csv"
state_file = "./output/template_map.txt"
#state = np.array([1,2,1,2])

def main():
    settings_obj = T6_PSI_settings.load_obj()
    state = np.zeros((settings_obj.NUM_REGIONS_X,settings_obj.NUM_REGIONS_Y))
    with open(state_file, 'r') as infile:
        for line in infile:
            data = re.findall(r'(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+((?:\d+\.?\d*|\w+))\s*',line)
            data2 = [float(i) for i in data[0]]
            x0,y0,x1,y1,temp =data2
            x = int(x0/(settings_obj.WIDTH_REGION*1e6))
            y = int(y0/(settings_obj.LENGTH_REGION*1e6))
            state[x][y] = temp
    #state = np.zeros((settings_obj.NUM_REGIONS_Y,settings_obj.NUM_REGIONS_X))
    state = state.reshape(settings_obj.NUM_REGIONS_X*settings_obj.NUM_REGIONS_Y)
    state = state.astype(int)
    current_map = np.genfromtxt(current_map_file, delimiter=',')
    current_map = (current_map) / settings_obj.VDD
    #print(state)
    #generate_IR_map(state,current_map)
    generate_IR_map_regionwise(state,current_map)

def generate_IR_map_regionwise(state,current_map):
    eq_obj = construct_eqn()
    settings_obj = T6_PSI_settings.load_obj()
    
    template_list = load_templates()
    max_drop = settings_obj.VDD * np.ones(len(state))
    voltage = np.zeros((0,3)) # location value tuple, (x,y,v)
    for y in range(settings_obj.NUM_REGIONS_Y):
        for x in range(settings_obj.NUM_REGIONS_X):
            n = y*settings_obj.NUM_REGIONS_Y + x
            template = state[n]
            regional_current, map_row = eq_obj.get_regional_current(
                current_map, x, y)
            template_obj = template_list[template]
            #g_start = template_obj.start
            J = eq_obj.create_J(template_obj,regional_current)
            #print(np.sum(J))
            G_orig = template_obj.get_G_mat()
            #print(G_orig.sum())
            #pprint(G_orig.sum(axis=0).tolist())
            #pprint(G_orig.sum(axis=1).tolist())
            G, J = eq_obj.add_vdd_to_G_J(J, template_obj)
            #pprint(G)
            #print(G.sum())
            J = sparse_mat.dok_matrix(J)
            solution = eq_obj.solve_ir(G, J)
            region_voltage = eq_obj.get_regional_voltage( template_obj, solution, x, y)
            voltage = np.append(voltage, region_voltage,axis =0)
            max_drop[n] = max(settings_obj.VDD - region_voltage.flatten())

    wc_ir = max(max_drop)
    #img.imsave('./output/IR_map.png', V_full)
    IR_drop = voltage
    IR_drop[:,2] = settings_obj.VDD - IR_drop[:,2]
    with open('./output/IR_drop.csv', 'wb') as outfile:
        np.savetxt(outfile,IR_drop,delimiter=',')
    with open('./output/IR_drop.rpt','w') as outfile:
        outfile.write("Worst case IR drop = %fV\n"%(wc_ir))
        if wc_ir > settings_obj.IR_DROP_LIMIT:
            outfile.write("Static IR drop specification VIOLATED")
        else:
            outfile.write("Static IR drop specification MET")

if __name__ == '__main__':
    main()
