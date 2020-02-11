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
from create_template import load_templates
from create_template import template
from create_template import node
from T6_PSI_settings import T6_PSI_settings
from construct_eqn import construct_eqn
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
    template_names_list = settings_obj.template_names_list
    state = [] 
    with open(state_file, 'r') as infile:
        for line in infile:
            data = re.findall(r'(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+((?:\d+\.?\d*|\w+))\s*',line)
            data2 = [float(data[0][i]) for i in range(len(data[0])-1)]
            x0,y0,x1,y1 =data2
            temp = data[0][4] 
            if temp in template_names_list:
                temp_num = template_names_list.index(temp)
            else:
                exit("error template name not found in database");
            x = int(x0/(settings_obj.WIDTH_REGION*1e6))
            y = int(y0/(settings_obj.LENGTH_REGION*1e6))
            val = [x0,x1,y0,y1,temp_num]
            state.append(val)
    current_map = np.genfromtxt(current_map_file, delimiter=',')
    #print(state)
    #generate_IR_map(state,current_map)
    generate_IR_map_regionwise(state,current_map)

def generate_IR_map_regionwise(state,current_map):
    eq_obj = construct_eqn()
    settings_obj = T6_PSI_settings.load_obj()
    
    template_list = load_templates()
    max_drop = settings_obj.VDD * np.ones(len(state))
    voltage = np.zeros((0,3)) # location value tuple, (x,y,v)
    for n,val in enumerate(state):
        template = state[n][4]
        x0,x1,y0,y1 = [int(i) for i in state[n][0:4]]
        regional_current, map_row = eq_obj.get_regional_current_from_coordinates(
            current_map, x0,x1,y0,y1)
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
        region_voltage = eq_obj.get_regional_voltage_from_coordinates( 
            template_obj, solution, x0,y0)
        voltage = np.append(voltage, region_voltage,axis =0)

    #img.imsave('./output/IR_map.png', V_full)
    IR_drop = voltage
    IR_drop[:,2] = settings_obj.VDD - IR_drop[:,2]
    wc_ir = max(IR_drop[:,2])
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
