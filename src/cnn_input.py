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
Created on Thu Mar 14 18:44:26 2019

@author:Vidya A Chhabria
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import tensorflow as tf
import numpy as np    # linear algebra
import pandas as pd    # data processing, CSV file I/O (e.g. pd.read_csv)
import json
from T6_PSI_settings import T6_PSI_settings
#from eliminate_templates import load_template_list


class cnn_input():
    def __init__(self):
        self.settings_obj = T6_PSI_settings.load_obj()
        #CLASSES = list(np.arange(N_CLASSES))
        dirname = self.settings_obj.template_file
        self.CLASSES = self.settings_obj.template_names_list
        
        #CLASSES = load_template_list()
        self.N_CLASSES = len(self.CLASSES)
        #settings_obj.NUM_TEMPLATES
        
        self.num_map_1d = 1
        self.MAP_SIZE_X = int(self.num_map_1d*self.settings_obj.WIDTH_REGION * 1e6)
        self.MAP_SIZE_Y = int(self.num_map_1d*self.settings_obj.LENGTH_REGION * 1e6)
        self.size_region_x = int(self.settings_obj.WIDTH_REGION * 1e6)
        self.size_region_y = int(self.settings_obj.LENGTH_REGION * 1e6)
        
        self.MAP_SIZE_1d = self.MAP_SIZE_X*self.MAP_SIZE_Y


    def load_and_preprocess_data(self,congestion_enabled):
        #if congestion_enabled ==1:
        #   normalization_file = self.settings_obj.checkpoint_dir+self.settings_obj.normalization_file
        #else:
        #   normalization_file = self.settings_obj.checkpoint_dir_wo_cong+self.settings_obj.normalization_file
        normalization_file = self.settings_obj.checkpoint_dir+'/'+self.settings_obj.normalization_file
        print("Preprocessing the input data")
        curr_train = np.array(
            pd.read_csv("output/train_currents.csv", header=None).values)
        curr_valid = np.array(
            pd.read_csv("output/val_currents.csv", header=None).values)
        curr_test = np.array(
            pd.read_csv("output/test_currents.csv", header=None).values)
        indices_train = pd.read_csv("output/train_template.csv",
                                    header=None).values
        indices_valid = pd.read_csv("output/val_template.csv",
                                    header=None).values
        indices_test = pd.read_csv("output/test_template.csv",
                                   header=None).values
    
        print("Preprocessing: data loaded")
        #normalizing paramters
        min_cur = np.amin(curr_train)
        max_cur = np.amax(curr_train)
        scl_cur = 1 / (max_cur - min_cur)
        #print("SCALING FACTORS\n")
        #print("min %e\n" % min_cur)
        #print("max %e\n" % max_cur)
        normalization= {}
        normalization['currents'] = {}
        normalization['currents']['max'] = max_cur
        normalization['currents']['min'] = min_cur
        print("Preprocessing: normalization parameters calculated and saved")
        with open(normalization_file, 'w') as outfile:
            json.dump(normalization, outfile, indent=4)
    
        curr_train = (curr_train - min_cur) * scl_cur
        curr_valid = (curr_valid - min_cur) * scl_cur
        curr_test = (curr_test - min_cur) * scl_cur
        print("Preprocessing: currents_normalized")
    
        template_train = np.zeros((indices_train.size, self.N_CLASSES))
        template_valid = np.zeros((indices_valid.size, self.N_CLASSES))
        template_test = np.zeros((indices_test.size, self.N_CLASSES))
    
        for n, c in enumerate(self.CLASSES):
            template_train[np.where(indices_train == n), n] = 1
            template_valid[np.where(indices_valid == n), n] = 1
            template_test[np.where(indices_test == n), n] = 1
    
        print("Preprocessing: Indices processed")
    
        num_train = curr_train.shape[0]
        num_valid = curr_valid.shape[0]
        num_test = curr_test.shape[0]

        print("Preprocessing: Completed")
        return curr_train, curr_valid, curr_test, template_train, \
            template_valid, template_test, num_train, num_valid, num_test
