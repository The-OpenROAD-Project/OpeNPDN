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
Created on Sun Dec 16 01:36:00 2018
This script is the main function which calls the functions to create G matrices
for all the possible templates and stores them. This ensures you do not have to
create it on the fly for every iteration
@author:Vidya A Chhabria
"""

import time
import sys
import os.path
import numpy as np
import itertools
from scipy import sparse
from template_construction import template_def
from T6_PSI_settings import T6_PSI_settings


def main():
    """ Main code which calls all the function to create the
    G matrices for all tempaltes"""
    start = time.time()
    settings_obj = T6_PSI_settings.load_obj()
    template_list = define_templates(settings_obj, generate_g=1)
    end = time.time()
    print("Creation time:%4.3f"%(end - start))
    dirname = settings_obj.template_file
    fname = dirname + "/refined_template_list.txt" 
    template_list_num = np.arange(len(template_list))
    np.savetxt(fname,template_list_num,fmt='%d')

    #print("\n")


def define_templates(settings_obj, generate_g):
    """ Solves the system of linear equations GV=J, to find V
        This function uses sparse matrix solver with umfpack
        Args:
            settings_obj: A dictionay which represents information in both the
            JSON files
            generate_g: A flag variable which decides whether to gernerate all
            the templates. Since this process is long, the flag is useful if the
            templets are already generated
        Returns:
            template_list: A list of objects of class template_def.
    """
    width_values = []
    res_per_l = []
    via_res = []
    dirs = []
    min_width = []
    pitches = []
    # Store a local copy of the variables from the settings_obj object from the
    # information in the JSON file
    for layer in settings_obj.PDN_layers_ids:
        attributes = settings_obj.LAYERS[layer]
        width_values.append(attributes['width'])
        min_width.append(attributes['min_width'])
        res_per_l.append(attributes['res'])
        dirs.append(attributes['direction'])
        pitches.append(attributes['pitch'])
    layer = settings_obj.TECH_LAYERS[0]
    via_res_1 = settings_obj.LAYERS[layer]['via_res']

    #TODO handle for generic names
    # Create the template with multiple layers based on the combination of
    # pitches of layers in the JSON
    for l in range(1, settings_obj.NUM_LAYERS ):
        layer = settings_obj.TECH_LAYERS[l]
        if layer in settings_obj.PDN_layers_ids:
            via_res.append(via_res_1)
            via_res_1 = settings_obj.LAYERS[layer]['via_res']
        else:
            via_res_1 += float(
                settings_obj.LAYERS[layer]['via_res'])

    width_values = np.array(width_values)
    res_per_l = np.array(res_per_l)
    via_res = np.array(via_res)
    # Set dir = 1 for those layers in the PDN which are veritcal
    dirs = np.array([(d == "V") for d in dirs]) * 1
    rho_values = res_per_l * min_width * 1e6
    # Setting the pitch values for every layer in the template
    pitch_values = np.zeros(
        (settings_obj.NUM_TEMPLATES, settings_obj.NUM_PDN_LAYERS))
    template_layers = []
    for p, layer_pitch in enumerate(pitches):
        num_layer_pitches = len(layer_pitch)
        if num_layer_pitches <= 0:
            print("ERROR: pitch of a PDN layer undefined")
            sys.exit()
        elif num_layer_pitches == 1:
            pitch_values[:, p] = layer_pitch[-1]
        else:
            #if num_layer_pitches == settings_obj.NUM_TEMPLATES:
            template_layers.append(p)
            #else:
            #    print(
            #        "ERROR: Layer %d does not have the corrent number of templates."
            #        % p, "Please check the template_definition.json file")
            #    sys.exit()
    #ranges = [range(len(pitches[x])) for x in template_layers]
    #template_num = 0
    #for pitch_idx in itertools.product(*ranges):
    #    print(pitch_idx)
    #    if(template_num >= settings_obj.NUM_TEMPLATES):
    #        print("ERROR: number of templates generated is greater than the number provided. Please check the template_definition.json file")
    #        sys.exit()
    #    for i in range(len(pitch_idx)):
    #        pitch_values[template_num,template_layers[i]] = pitches[template_layers[i]][pitch_idx[i]]
    #    template_num +=1

    for template_num in range(settings_obj.NUM_TEMPLATES):
        for p in template_layers:
            pitch_values[template_num, p] = pitches[p][template_num]
    template_num +=1

    if template_num != settings_obj.NUM_TEMPLATES :
        print(template_num)
        print(
            "ERROR: number of templates generated does not match number provided.",
            "Please check the template_definition.json file")
        sys.exit()
    template_list = []
    init_offset = np.zeros((pitch_values.shape[0], pitch_values.shape[1]))
    # Call to the template definition which calls the function that creates G
    for temp_num, pitch_template in enumerate(pitch_values):
        template_obj = template_def(settings_obj.NUM_PDN_LAYERS, pitch_template,
                                    width_values, rho_values, via_res, dirs,
                                    settings_obj.LENGTH_REGION,
                                    settings_obj.WIDTH_REGION)
        init_offset[temp_num, :] = template_obj.init_offset.flatten()
    max_offset = np.amax(init_offset, axis=0)

    for temp_num, pitch_template in enumerate(pitch_values):
        dirname = settings_obj.template_file
        if not os.path.exists(dirname):
            print("#########################################################")
            print("CREATING TEMPLATE DIRECTORY")
            print("#########################################################")
        fname = dirname + "/template_obj_%d.npz" % temp_num
        template_obj = template_def(settings_obj.NUM_PDN_LAYERS, pitch_template,
                                    width_values, rho_values, via_res, dirs,
                                    settings_obj.LENGTH_REGION,
                                    settings_obj.WIDTH_REGION)
        grid_arr = np.zeros(dirs.shape[0])
        grid_arr[dirs == 0] = template_obj.ypitch
        grid_arr[dirs == 1] = template_obj.xpitch
        pitch_nodes = np.array(pitch_template) / grid_arr
        init_offset[temp_num, init_offset[temp_num, :] ==
                    0] = 1    # handle 0/0 condition
        template_obj.offset = np.remainder(max_offset, pitch_nodes)
        # generate g is a user flag which is one when properties of template
        # change examples include
        if generate_g == 1:
            print("Creating Template %d " % temp_num)
            template_obj.create_G()
            template_list.append(template_obj)
            G = sparse.dok_matrix.tocsc(template_obj.G)
           # print("Shape of the matrix", G.get_shape())
            with open(fname, 'wb') as config_dictionary_file:
                sparse.save_npz(config_dictionary_file, G)
        else:
            if not os.path.isfile(fname):
                print("######################################################")
                print("TEMPLATE %d NOT FOUND. Run create_template.py" %
                      temp_num)
                print("######################################################")
            else:
                pass
                #print("Loading Template %d from file. If you have" % temp_num,
                #      "changed the json file ensure to run create_template.py")
            with open(fname, 'rb') as config_dictionary_file:
                G = sparse.load_npz(config_dictionary_file)
            template_obj.G = G.todok()
            template_list.append(template_obj)
    return template_list

if __name__ == '__main__':
    main()
