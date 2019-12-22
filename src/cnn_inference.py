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

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 22:10:01 2019
This script runs the inference flow using the trained optimized CNN to predict
the class for a given testcase
@author:Vidya A Chhabria
"""

import json
import numpy as np
import tensorflow as tf
from cnn_input import cnn_input
from cnn import cnn
from pprint import pprint
from T6_PSI_settings import T6_PSI_settings
import os
import sys

class cnn_inference():
    def __init__(self):
        #TODO take in the settings file
        self.cnn_input_obj = cnn_input()
        self.cnn_obj = cnn()

        self.power_map_file = "./work/current_map_processed.csv"
        self.cong_map_file = "./output/congestion_map.csv"
        #congestion_map_file = "./work/congestion_processed.csv"
        self.settings_obj = T6_PSI_settings.load_obj()
        
        if (len(sys.argv)>1 and sys.argv[1] == "no_congestion"):
            self.congestion_enabled = 0 
        else: 
            self.congestion_enabled = 1 
        
        
        if self.congestion_enabled ==1:
            self.checkpoint_dir = self.settings_obj.checkpoint_dir
        else:
            self.checkpoint_dir = self.settings_obj.checkpoint_dir_wo_cong
        normalization_file = self.checkpoint_dir+self.settings_obj.normalization_file
        # Golden template numbers needed for comparison
        # This corresponds only to the first current map
        #indices = np.zeros((self.settings_obj.NUM_REGIONS_X * self.settings_obj.NUM_REGIONS_Y))
        self.test_size = self.settings_obj.NUM_REGIONS_X * self.settings_obj.NUM_REGIONS_Y
        # Hard coded after seeing training data, need to fix
        with open(normalization_file) as f:
            norm_data = json.load(f)
        min_cur = norm_data['currents']['min']
        max_cur = norm_data['currents']['max']
        self.scl_cur = 1 / (max_cur - min_cur)
        
        if self.congestion_enabled == 1:
            min_cong = norm_data['congestion']['min']
            max_cong = norm_data['congestion']['max']
            self.scl_cong = 1 / (max_cong - min_cong)
        
        self.template_map_file = "./output/template_map.txt"
    
    
    def eval_once(self,currents_testcase, congestion_testcase, template_testcase):
        """ This function loads the checkpoint of the trained CNN and takes in the
        testcase currents to predict the output and check the testcase accuracy
        Args:
            currents_testcase: Matrix corresponding to the current map of the
                               of the testcase being evaluated
            template_testcase: An array of the actual templates of the testcase
        """
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.logging.set_verbosity(tf.logging.ERROR)
        with tf.Graph().as_default():
            #with tf.name_scope('data'):
            maps = tf.placeholder(tf.float32,
                                  [None, self.cnn_input_obj.MAP_SIZE_1d],
                                  name="X_placeholder")
            labels = tf.placeholder(tf.float32,
                                    [None, self.cnn_input_obj.N_CLASSES],
                                    name="Y_placeholder")
            if self.congestion_enabled == 1:
                cong = tf.placeholder(tf.float32,
                                  [None, self.cnn_input_obj.MAP_SIZE_1d],
                                  name="C_placeholder")
            else:
                cong = tf.constant(0)
            logits = self.cnn_obj.inference(maps,cong,self.congestion_enabled)
            pred = self.cnn_obj.prediction(logits)
            accuracy = self.cnn_obj.accuracy(logits, labels)
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                if ckpt and ckpt.model_checkpoint_path:
                    # Restores from checkpoint
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    print('No checkpoint file found')
                    return
    
                X_batch = currents_testcase
                Y_batch = template_testcase
                if self.congestion_enabled == 1:
                    C_batch = congestion_testcase
                #TODO handle size mismatches
                    pred_acc,predict = sess.run([accuracy,pred],
                                     feed_dict={
                                         maps: X_batch,
                                         cong: C_batch,
                                         labels: Y_batch
                                     })
                else:
                    pred_acc,predict = sess.run([accuracy,pred],
                                     feed_dict={
                                         maps: X_batch,
                                         labels: Y_batch
                                     })
                #eval_acc = 100 * pred_acc / self.test_size
                #print("Validation Accuracy {0}".format(eval_acc))
                #print("Predicted templates\n")
                #pprint(predict)
                print("INFO: Saving optimized PDN in template_map.txt")
                region=0;
                size_region_x = (self.settings_obj.WIDTH_REGION * 1e6)
                size_region_y = (self.settings_obj.LENGTH_REGION * 1e6)
                with open(self.template_map_file,'w') as outfile:
                    for y in range(self.settings_obj.NUM_REGIONS_Y):
                        for x in range(self.settings_obj.NUM_REGIONS_X):
                            x0 = x*size_region_x
                            x1 = (x+1)*size_region_x
                            y0 = y*size_region_y
                            y1 = (y+1)*size_region_y
                            #TODO update to real name
                            template = predict[region]
                            outfile.write("%5.1f %5.1f %5.1f %5.1f %d\n"%(
                                    x0,y0,x1,y1,template))
                            region = region+1
    
    
    def process_testcase(self):
        """ This function preprocess the testcase data to make it into a input that
        is readale by the CNN. It creates a current map from a power report DEF and
        LEF. It then normalizes the data based on the mean and standard deviation of
        the training set.
        Returns:
             currents and templates: The inputs in the propsed form.
        """
    #    with tf.Graph().as_default():
    #        #with tf.name_scope('data'):
    #        maps = tf.pl
        self.settings_obj = T6_PSI_settings.load_obj()
        size_region_x = int(self.settings_obj.WIDTH_REGION * 1e6)
        size_region_y = int(self.settings_obj.LENGTH_REGION * 1e6)
        curr_testcase = np.zeros(
            (self.settings_obj.NUM_REGIONS_X * self.settings_obj.NUM_REGIONS_Y,
             3*3*size_region_x * size_region_y))
    #    cong_testcase = np.genfromtxt(congestion_map_file, delimiter=',')
        currents = np.genfromtxt(self.power_map_file, delimiter=',')
        if self.congestion_enabled == 1:
            cong_testcase = np.zeros((self.settings_obj.NUM_REGIONS_X * self.settings_obj.NUM_REGIONS_Y,3*3*size_region_x * size_region_y))
            congestion = np.genfromtxt(self.cong_map_file, delimiter=',')
        currents = (currents) / self.settings_obj.VDD
        n=0
        for y in range(self.settings_obj.NUM_REGIONS_Y):
            for x in range(self.settings_obj.NUM_REGIONS_X):
                current_region = np.zeros((3*size_region_x,3*size_region_y))
                if self.congestion_enabled == 1:
                    congestion_region = np.zeros((3*size_region_x,3*size_region_y))
    
                if  self.settings_obj.NUM_REGIONS_Y == 1:
                    y_start = 0 
                    y_end =  y_start+size_region_y
                    y_reg_start = size_region_y
                    y_reg_end =  y_reg_start+size_region_y
                elif y == 0 :
                    y_start = 0
                    y_end =  y_start+2*size_region_y
                    y_reg_start = size_region_y
                    y_reg_end =  y_reg_start+2*size_region_y
                elif y == self.settings_obj.NUM_REGIONS_Y-1:
                    y_start = (y-1)*size_region_y
                    y_end =  y_start+2*size_region_y
                    y_reg_start = 0
                    y_reg_end =  y_reg_start+2*size_region_y
                else:
                    y_start = (y-1)*size_region_y
                    y_end =  y_start+3*size_region_y
                    y_reg_start = 0
                    y_reg_end =  y_reg_start+3*size_region_y
                if  self.settings_obj.NUM_REGIONS_X == 1:
                    x_start = 0 
                    x_end =  x_start+size_region_x
                    x_reg_start = size_region_x
                    x_reg_end =  x_reg_start+size_region_x
                elif x == 0 :
                    x_start = 0
                    x_end =  x_start+2*size_region_x
                    x_reg_start = size_region_x
                    x_reg_end =  x_reg_start+2*size_region_x
                elif x == self.settings_obj.NUM_REGIONS_X-1:
                    x_start = (x-1)*size_region_x
                    x_end =  x_start+2*size_region_x
                    x_reg_start = 0
                    x_reg_end =  x_reg_start+2*size_region_x
                else:
                    x_start = (x-1)*size_region_x
                    x_end =  x_start+3*size_region_x
                    x_reg_start = 0
                    x_reg_end =  x_reg_start+3*size_region_x
    
                current_region[x_reg_start:x_reg_end,y_reg_start:y_reg_end] = (
                            currents[x_start:x_end,y_start:y_end])
                curr_testcase[n] = current_region.reshape(-1)*self.scl_cur
    
                if self.congestion_enabled == 1:
                    congestion_region[x_reg_start:x_reg_end,y_reg_start:y_reg_end] = np.mean(
                                congestion[x_start:x_end,y_start:y_end])
                    
                    cong_testcase[n] = congestion_region.reshape(-1)*self.scl_cong
                else: 
                    cong_testcase = 0
                n =n +1
        template_testcase = np.zeros((self.test_size, self.cnn_input_obj.N_CLASSES))
        #for n, c in enumerate(self.cnn_input_obj.CLASSES):
        #    template_testcase[np.where(indices == c), n] = 1
    
        return curr_testcase, cong_testcase, template_testcase
    

if __name__ == "__main__":
    cnn_inf_obj = cnn_inference()
    curr, cong,temp = cnn_inf_obj.process_testcase()
    cnn_inf_obj.eval_once(curr, cong, temp)
