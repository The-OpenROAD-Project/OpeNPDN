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
from eliminate_templates import load_template_list


settings_obj = T6_PSI_settings()
#CLASSES = list(np.arange(N_CLASSES))
CLASSES = load_template_list()
N_CLASSES = len(CLASSES)
#settings_obj.NUM_TEMPLATES

num_map_1d = 3
MAP_SIZE_X = int(num_map_1d*settings_obj.WIDTH_REGION * 1e6)
MAP_SIZE_Y = int(num_map_1d*settings_obj.LENGTH_REGION * 1e6)
size_region_x = int(settings_obj.WIDTH_REGION * 1e6)
size_region_y = int(settings_obj.LENGTH_REGION * 1e6)

MAP_SIZE_1d = MAP_SIZE_X*MAP_SIZE_Y


def load_and_preprocess_data(congestion_enabled):
    if congestion_enabled ==1:
       normalization_file = settings_obj.checkpoint_dir+settings_obj.normalization_file
    else:
       normalization_file = settings_obj.checkpoint_dir_wo_cong+settings_obj.normalization_file
    print("Preprocessing the input data")
    curr_train = np.array(
        pd.read_csv("output/CNN_train_currents.csv", header=None).values)
    curr_valid = np.array(
        pd.read_csv("output/CNN_val_currents.csv", header=None).values)
    curr_test = np.array(
        pd.read_csv("output/CNN_test_currents.csv", header=None).values)
    if(congestion_enabled == 1):
        pre_cong_train = np.array(
            pd.read_csv("output/CNN_train_congest.csv", header=None).values)
        pre_cong_valid = np.array(
            pd.read_csv("output/CNN_val_congest.csv", header=None).values)
        pre_cong_test = np.array(
            pd.read_csv("output/CNN_test_congest.csv", header=None).values)

    print("Preprocessing: data loaded")
    #normalizing paramters
    min_cur = np.amin(curr_train)
    max_cur = np.amax(curr_train)
    scl_cur = 1 / (max_cur - min_cur)
    if(congestion_enabled == 1):
        min_cong = np.amin(pre_cong_train)
        max_cong = np.amax(pre_cong_train)
        scl_cong = 1 / (max_cong - min_cong)
    else:
        min_cong = 0
        max_cong = 1
        scl_cong = 0
    #print("SCALING FACTORS\n")
    #print("min %e\n" % min_cur)
    #print("max %e\n" % max_cur)
    normalization= {}
    normalization['currents'] = {}
    normalization['congestion'] = {}
    normalization['currents']['max'] = max_cur
    normalization['currents']['min'] = min_cur
    normalization['congestion']['max'] = max_cong
    normalization['congestion']['min'] = min_cong
    print("Preprocessing: normalization parameters calculated and saved")
    with open(normalization_file, 'w') as outfile:
        json.dump(normalization, outfile, indent=4)

    curr_train = (curr_train - min_cur) * scl_cur
    curr_valid = (curr_valid - min_cur) * scl_cur
    curr_test = (curr_test - min_cur) * scl_cur
    print("Preprocessing: currents_normalized")
    if(congestion_enabled == 1):
        pre_cong_train = (pre_cong_train - min_cong) * scl_cong
        pre_cong_valid = (pre_cong_valid - min_cong) * scl_cong
        pre_cong_test =  (pre_cong_test -  min_cong) * scl_cong

        print("Preprocessing: congestion_normalized")
        cong_train = np.zeros((curr_train.shape[0],curr_train.shape[1]))
        for i,cong in enumerate(pre_cong_train):
            cong_map = np.zeros((MAP_SIZE_X,MAP_SIZE_Y))
            for n,c in enumerate(cong):
                x  = n % num_map_1d
                y  = int(n / num_map_1d)
                x_str = x*size_region_x
                x_stp = (x+1)*size_region_x
                y_str = y*size_region_y
                y_stp = (y+1)*size_region_y
                cong_map[x_str:x_stp,y_str:y_stp] = c*np.ones((size_region_x,size_region_y))
            cong_train[i,:] = cong_map.reshape(-1)
        cong_valid = np.zeros((curr_valid.shape[0],curr_valid.shape[1]))
        for i,cong in enumerate(pre_cong_valid):
            cong_map = np.zeros((MAP_SIZE_X,MAP_SIZE_Y))
            for n,c in enumerate(cong):
                x  = n % num_map_1d
                y  = int(n / num_map_1d)
                x_str = x*size_region_x
                x_stp = (x+1)*size_region_x
                y_str = y*size_region_y
                y_stp = (y+1)*size_region_y
                cong_map[x_str:x_stp,y_str:y_stp] = c*np.ones((size_region_x,size_region_y))
            cong_valid[i,:] = cong_map.reshape(-1)
        cong_test = np.zeros((curr_test.shape[0],curr_test.shape[1]))
        for i,cong in enumerate(pre_cong_test):
            cong_map = np.zeros((MAP_SIZE_X,MAP_SIZE_Y))
            for n,c in enumerate(cong):
                x  = n % num_map_1d
                y  = int(n / num_map_1d)
                x_str = x*size_region_x
                x_stp = (x+1)*size_region_x
                y_str = y*size_region_y
                y_stp = (y+1)*size_region_y
                cong_map[x_str:x_stp,y_str:y_stp] = c*np.ones((size_region_x,size_region_y))
            cong_test[i,:] = cong_map.reshape(-1)
            
        print("Preprocessing: congestion_processed")
    else:
        cong_train =0 
        cong_test =0 
        cong_valid =0

    indices_train = pd.read_csv("output/CNN_train_template.csv",
                                header=None).values
    indices_valid = pd.read_csv("output/CNN_val_template.csv",
                                header=None).values
    indices_test = pd.read_csv("output/CNN_test_template.csv",
                               header=None).values
    template_train = np.zeros((indices_train.size, N_CLASSES))
    template_valid = np.zeros((indices_valid.size, N_CLASSES))
    template_test = np.zeros((indices_test.size, N_CLASSES))

    for n, c in enumerate(CLASSES):
        template_train[np.where(indices_train == c), n] = 1
        template_valid[np.where(indices_valid == c), n] = 1
        template_test[np.where(indices_test == c), n] = 1

    print("Preprocessing: Indices processed")

    num_train_cur = curr_train.shape[0]
    num_valid_cur = curr_valid.shape[0]
    num_test_cur = curr_test.shape[0]
    if(congestion_enabled == 1):
        num_train_cong = cong_train.shape[0]
        num_valid_cong = cong_valid.shape[0]
        num_test_cong =  cong_test.shape[0]
        if (num_train_cur == num_train_cong and num_valid_cur == num_valid_cong and
        num_test_cur == num_test_cong):
            num_train = num_train_cur
            num_valid = num_valid_cur
            num_test = num_test_cur
        else:
            print("ERROR: Current and congestion input training data are not of the same size")
            exit()
    else:
        num_train = num_train_cur
        num_valid = num_valid_cur
        num_test = num_test_cur

    print("Preprocessing: Completed")
    return curr_train, curr_valid, curr_test, cong_train, cong_valid, cong_test, \
        template_train, template_valid, \
        template_test, num_train, num_valid, num_test
