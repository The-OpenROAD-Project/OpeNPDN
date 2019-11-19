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
from create_template import define_templates
from T6_PSI_settings import T6_PSI_settings
from simulated_annealer import simulated_annealer
from construct_eqn import construct_eqn
from scipy import sparse as sparse_mat
import matplotlib.image as img
import math
import time
import re
from scipy import interpolate



current_map_file = "./work/current_map_processed.csv"
state_file = "./output/template_map.txt"
#state = np.array([1,2,1,2])

def main():
    # Read the json user input file and the current maps that need to be run
    # taken as an argument from the scripts
    #settings_obj = T6_PSI_settings()
    pass
#    T_init = 70
#    T_final = 0.0005
#    alpha_temp = 0.95
#    num_moves_per_step = 5
#
#    state = np.zeros((1, settings_obj.NUM_REGIONS))
#    e = np.zeros(1)
#    max_drop = np.zeros((1, settings_obj.NUM_REGIONS))
#    template_list = define_templates(settings_obj, generate_g=0)
#    for i in range(1):
#        print(i)
#        power_map_file = settings_obj.map_dir + "current_map_%d.csv" % (
#            i + map_start)
#        currents = np.genfromtxt(power_map_file, delimiter=',')
#
#        all_templates = list(range(settings_obj.NUM_TEMPLATES))
#        init_state = np.zeros(settings_obj.NUM_REGIONS, int)
#
#        if len(init_state) != settings_obj.NUM_REGIONS:
#            print("please check the length of init state")
#        pdn_opt = simulated_annealer(init_state, T_init, T_final, alpha_temp,
#                                     num_moves_per_step, currents)
#        state[i, :], e[i], max_drop[i, :] = pdn_opt.sim_anneal(
#            all_templates, template_list)
#    with open(
#            settings_obj.parallel_run_dir + 'max_drop_%d_to_%d.csv' %
#        (map_start, map_start + num_maps - 1), 'w') as outfile:
#        np.savetxt(outfile, max_drop, delimiter=',', fmt='%f')
#    with open(
#            settings_obj.parallel_run_dir + 'state_%d_to_%d.csv' %
#        (map_start, map_start + num_maps - 1), 'w') as outfile:
#        np.savetxt(outfile, state, delimiter=',', fmt='%d')
#    with open(
#            settings_obj.parallel_run_dir + 'energy_%d_to_%d.csv' %
#        (map_start, map_start + num_maps - 1), 'w') as outfile:
#        np.savetxt(outfile, e, delimiter=',', fmt='%f')

def generate_IR_map_regionwise(state,current_map):
    eq_obj = construct_eqn()
    settings_obj = T6_PSI_settings()
    
    template_list = define_templates(settings_obj, generate_g=0)
    max_drop = settings_obj.VDD * np.ones(len(state))
    for y in range(settings_obj.NUM_REGIONS_Y):
        for x in range(settings_obj.NUM_REGIONS_X):
            n = y*settings_obj.NUM_REGIONS_Y + x
            template = state[n]
            regional_current, map_row = eq_obj.get_regional_current(
                current_map, x, y)
            template_obj = template_list[template]
            g_start = template_obj.start
            G = template_obj.G
            J = eq_obj.create_J(regional_current, template_obj)
            G, J = eq_obj.add_vdd_to_G_J(G, J, template_obj, 0)
            J = sparse_mat.dok_matrix(J)
            solution = eq_obj.solve_ir(G, J)
            bot = g_start[0]    #M1 is shape -6
            top = g_start[1]
            V = solution[int(bot):int(top)]
            dimx = template_list[template].num_x
            dimy = template_list[template].num_y
            max_drop[n] = max(settings_obj.VDD - V)
            #print("region %d template %d"%(n,template))
            V = V.reshape((dimx,dimy))
            if x == 0:
                V_row = V.T
            else:
                V_row = np.vstack((V_row, V.T))
        if y == 0:
            V_full = V_row
        else:
            V_full = np.hstack((V_full,V_row))
#            if(n ==0 ):
#                V_full = V.T
#            else:
#                V_full = np.vstack((V_full,V.T))
#            J_map = J[int(bot):int(top)]
#            J_map = J_map.todense()
#            if(n ==0 ):
#                J_full = J_map.T
#            else:
#                J_full = np.vstack((J_full,J_map.T))
    wc_ir = max(max_drop)
    img.imsave('./output/IR_map.png', V_full)
    with open('./output/IR_drop.csv', 'wb') as outfile:
        np.savetxt(outfile,(settings_obj.VDD-V_full),delimiter=',')
    with open('./output/IR_drop.rpt','w') as outfile:
        outfile.write("Worst case IR drop = %fV\n"%(wc_ir))
        if wc_ir > settings_obj.IR_DROP_LIMIT:
            outfile.write("Static IR drop specification VIOLATED")
        else:
            outfile.write("Static IR drop specification MET")
#        with open('J_map.csv', 'w') as outfile:
#            np.savetxt(outfile,J_full,delimiter=',')

def interpolate_data(V,dimx,dimy):
    settings_obj = T6_PSI_settings()
    width = int(settings_obj.WIDTH_REGION*1e6)
    height = int(settings_obj.LENGTH_REGION*1e6)
    #print("width height inside %d %d"%(width,height))
    #print("width height xy inside %d %d"%(dimx,dimy))

    X = np.linspace(0,width,dimx)
    Y = np.linspace(0,height,dimy)
    
    x,y = np.meshgrid(Y,X) # generates no2*no1 array
    
    #f = interpolate.interp2d(x,y,V,kind='linear')
    
    Xnew = np.linspace(0,width,width)
    Ynew = np.linspace(0,height,height)
    Xng,Yng = np.meshgrid(Ynew,Xnew)
    #Vnew = f(Xnew,Ynew)

    Vnew = interpolate.griddata( np.array([x.ravel(), y.ravel()]).T, V.ravel(), 
                        (Xng, Yng), method='cubic')

    #print(Vnew.shape)
    return Vnew    

def extrapolate_data(V,dimx,dimy):
    settings_obj = T6_PSI_settings()
    width_in = int(settings_obj.NUM_REGIONS_X*settings_obj.WIDTH_REGION*1e6)
    height_in = int(settings_obj.NUM_REGIONS_Y*settings_obj.LENGTH_REGION*1e6)

    X = np.linspace(0,width_in, width_in )
    Y = np.linspace(0,height_in,height_in)
    
    x,y = np.meshgrid(Y,X)
    
    Xnew = np.linspace(0,dimx,dimx)
    Ynew = np.linspace(0,dimy,dimy)
    Xng,Yng = np.meshgrid(Ynew,Xnew)
    #Vnew = f(Xnew,Ynew)

    #Vnew = interpolate.griddata( np.array([x.ravel(), y.ravel()]).T, V.ravel(), 
    #                    (Xng, Yng), method='cubic')
    Vnew_N = interpolate.griddata( np.array([x.ravel(), y.ravel()]).T, V.ravel(), 
                        (Xng, Yng), method='nearest')
    #extapolate the ends to the nearest 
    #Vnew[np.isnan(Vnew)] = Vnew_N[np.isnan(Vnew)]
    Vnew = Vnew_N
    return Vnew    


def generate_IR_map(state,current_map):
    eq_obj = construct_eqn()
    settings_obj = T6_PSI_settings()
    
    template_list = define_templates(settings_obj, generate_g=0)
    max_drop = settings_obj.VDD * np.ones(len(state))

    s1 =time.time()
    G,J, template_start = eq_obj.create_G_J (state, current_map,template_list)
    e1 =time.time()
    solution =eq_obj.solve_ir(G,J)
    e2 =time.time()

    print("INFO: Solving for static IR drop")
    for n,template in enumerate(state):
        g_start = template_list[template].start
        dimx = template_list[template].num_x
        dimy = template_list[template].num_y
        #print("dim x y %d %d"%(dimx,dimy))
        bot = g_start[0]    #M1 is shape -6
        top = g_start[1]
        V = solution[int(template_start[n]+bot):int(template_start[n]+top)]
        max_drop[n] = max(settings_obj.VDD - V)
        V = V.reshape((dimx,dimy),order='F')
        #print("shape of V %d %d"%(V.shape))
        #print("dimx dim y %d %d"%(dimx,dimy))
        V = interpolate_data(V,dimx,dimy)
        #print("shape of V %d %d"%(V.shape))
        if n % settings_obj.NUM_REGIONS_X == 0:
            V_row = V
            #print("shape of V row %d %d"%(V_row.shape))
            if n % settings_obj.NUM_REGIONS_X == settings_obj.NUM_REGIONS_X -1:
                if int(n / settings_obj.NUM_REGIONS_X) == 0:
                    V_full = V_row
                    #print("shape of V full %d %d"%(V_full.shape))
                else:
                    V_full = np.hstack((V_full,V_row))
        elif n % settings_obj.NUM_REGIONS_X == settings_obj.NUM_REGIONS_X -1:
            V_row = np.vstack((V_row, V))
            if int(n / settings_obj.NUM_REGIONS_X) == 0:
                V_full = V_row
            else:
                V_full = np.hstack((V_full,V_row))
        else:
            V_row = np.vstack((V_row, V))
    chip_dimx,chip_dimy = current_map.shape         
    #print("shape of curr %d %d"%(current_map.shape))
    #print("shape of curr calc %d %d"%(chip_dimx,chip_dimy))
    #print("shape of V_full %d %d"%(V_full.shape))
    V_full = extrapolate_data(V_full,chip_dimx,chip_dimy)
    #print("shape of V_full %d %d"%(V_full.shape))
    #print("INFO: Saving IR map report")
    #print(V_full.shape)
    wc_ir = max(max_drop)
    img.imsave('./output/IR_map.png', np.flipud(V_full.T))
    with open('./output/IR_drop.csv', 'wb') as outfile:
        np.savetxt(outfile,(settings_obj.VDD-V_full),delimiter=',')
    with open('./output/IR_drop.rpt','w') as outfile:
        outfile.write("Worst case IR drop = %fV\n"%(wc_ir))
        if wc_ir > settings_obj.IR_DROP_LIMIT:
            outfile.write("Static IR drop specification VIOLATED")
        else:
            outfile.write("Static IR drop specification MET")




if __name__ == '__main__':
    settings_obj = T6_PSI_settings()
    state = np.zeros((settings_obj.NUM_REGIONS_X,settings_obj.NUM_REGIONS_Y))
    with open(state_file, 'r') as infile:
        for line in infile:
            data = re.findall(r'Region x = (\d+) y = (\d+), template = (\d+)',line);
            data2 = [int(i) for i in data[0]]
            x,y,temp = data2
            assert x<settings_obj.NUM_REGIONS_X and x>=0, (
            "Index x in template map.txt is not within the number of regions defined in template_definition.json ")
            assert y<settings_obj.NUM_REGIONS_Y and y>=0, (
            "Index y in template map.txt is not within the number of regions defined in template_definition.json ")
            state[x][y] = temp
    #state = np.zeros((settings_obj.NUM_REGIONS_Y,settings_obj.NUM_REGIONS_X))
    state = state.reshape(settings_obj.NUM_REGIONS_X*settings_obj.NUM_REGIONS_Y)
    state = state.astype(int)
    current_map = np.genfromtxt(current_map_file, delimiter=',')
    current_map = (current_map) / settings_obj.VDD
    #print(state)
    generate_IR_map(state,current_map)
    #generate_IR_map_regionwise(state,current_map)
    main()
