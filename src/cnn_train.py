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
Created on Wed May  8 22:10:01 2019

@author:Vidya A Chhabria
"""

import tensorflow as tf
from cnn_input import cnn_input
from cnn import cnn
from T6_PSI_settings import T6_PSI_settings
import time
from tqdm import tqdm
import os
import sys

cnn_input_obj = cnn_input()
cnn_obj = cnn()
settings_obj = T6_PSI_settings.load_obj()
N_EPOCHS = settings_obj.N_EPOCHS
SKIP_STEP = 10 * 512 / cnn_obj.BATCH_SIZE
SAVE_STEP = 30 * 512 / cnn_obj.BATCH_SIZE
save_model = True


def train(congestion_enabled):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    #print "VERSION"
    #tf.__version__sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.reset_default_graph()
    #with tf.device("/gpu:0"):
    if congestion_enabled == 1:
        checkpoint_file = settings_obj.checkpoint_dir+settings_obj.checkpoint_file
    else:
        checkpoint_file = settings_obj.checkpoint_dir_wo_cong+settings_obj.checkpoint_file
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    tf.__version__sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    with tf.Graph().as_default():
        maps = tf.placeholder(tf.float32,
                              [None, cnn_input_obj.MAP_SIZE_1d],
                              name="X_placeholder")
        if(congestion_enabled ==1):
            cong = tf.placeholder(tf.float32,
                              [None, cnn_input_obj.MAP_SIZE_1d],
                              name="C_placeholder")
        else:
            cong = tf.constant(0)
        labels = tf.placeholder(tf.float32,
                                [None, cnn_input_obj.N_CLASSES],
                                name="Y_placeholder")

        global_step = tf.Variable(0,
                                  dtype=tf.int32,
                                  trainable=False,
                                  name='global_step')

        
        logits = cnn_obj.inference(maps,cong,congestion_enabled)
        loss = cnn_obj.loss(logits, labels)
        optimizer = cnn_obj.train(loss, global_step)
        accuracy = cnn_obj.accuracy(logits, labels)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            start_time = time.time()
            initial_step = global_step.eval()
            start_read = time.time()
            (curr_train, curr_valid, curr_test, cong_train, cong_valid,
            cong_test,template_train, template_valid, template_test, num_train,
            num_valid, num_test) = cnn_input_obj.load_and_preprocess_data(congestion_enabled)
            n_batches_train = int(num_train / cnn_obj.BATCH_SIZE)
            total_loss = 0.0
            total_loss10 = 0.0
            trn_btch_str = 0
            trn_btch_end = trn_btch_str + cnn_obj.BATCH_SIZE
            epoch_num = 1
            start_train = time.time()
            correct_preds_train =0
            for epcoh_num in tqdm(range(N_EPOCHS),desc='EPOCH NUM', total=N_EPOCHS):
                for trn_bat in range(0, n_batches_train ):
                    index = initial_step+epoch_num*n_batches_train+trn_bat
                    X_batch = curr_train[trn_btch_str:trn_btch_end, :]
                    Y_batch = template_train[trn_btch_str:trn_btch_end, :]
                    if(congestion_enabled ==1):
                        C_batch = cong_train[trn_btch_str:trn_btch_end, :]
                        correct_preds_batch,_, loss_batch = sess.run([accuracy,optimizer, loss],
                                         feed_dict={
                                             maps: X_batch,
                                             cong: C_batch,
                                             labels: Y_batch
                                             })
                    else:
                        correct_preds_batch,_, loss_batch = sess.run([accuracy,optimizer, loss],
                                         feed_dict={
                                             maps: X_batch,
                                             labels: Y_batch
                                             })
                    total_loss += loss_batch
                    total_loss10 += loss_batch

                    correct_preds_train = correct_preds_batch + correct_preds_train
                    if (index + 1) % 10 == 0:
                        print("epoch_num: %d elapsed_time = %f"%(int(index/n_batches_train) ,time.time()-start_time))
                        print('Average loss at step {}: {:5.5f}'.format(
                            index + 1, total_loss10 / 10))
                        total_loss10 = 0.0
                    if (index + 1) % SKIP_STEP == 0:
                        print("epoch_num: %d elapsed_time = %f"%(int(index/n_batches_train) ,time.time()-start_time))
                        print('Average loss at step {}: {:5.5f}'.format(
                            index + 1, total_loss / SKIP_STEP))
                        total_loss = 0.0
                        train_acc = 100*correct_preds_train/(cnn_obj.BATCH_SIZE*SKIP_STEP)
                        correct_preds_train =0
                        print("training Accuracy {0}".format(train_acc))

                        X_batch = curr_valid[0:num_valid, :]
                        Y_batch = template_valid[0:num_valid, :]
                        if(congestion_enabled ==1):
                            C_batch = cong_valid[0:num_valid, :]
                            correct_preds_batch, = sess.run([accuracy],
                                                        feed_dict={
                                                            maps: X_batch,
                                                            cong: C_batch,
                                                            labels: Y_batch
                                                        })
                        else:
                            correct_preds_batch, = sess.run([accuracy],
                                                        feed_dict={
                                                            maps: X_batch,
                                                            labels: Y_batch
                                                        })
                        valid_acc = 100 * correct_preds_batch / num_valid
                        print("Validation Accuracy {0}".format(valid_acc))

                    if ((index + 1) % SAVE_STEP == 0 and save_model == True):
                        print("SAVING CHECKPOINT")
                        saver.save(sess, checkpoint_file, index)
                    trn_btch_str, trn_btch_end, epoch_num = get_next_batch(
                        trn_btch_str, trn_btch_end, epoch_num, num_train)
            print("Optimization Finished!")
            print("SAVING CHECKPOINT")
            saver.save(sess, checkpoint_file, index)
            X_batch = curr_test[0:num_test, :]
            Y_batch = template_test[0:num_test, :]
            if(congestion_enabled ==1):
                C_batch = cong_test[0:num_test, :]
                correct_preds_batch, = sess.run([accuracy],
                                            feed_dict={
                                                maps: X_batch,
                                                cong: C_batch,
                                                labels: Y_batch
                                            })
            else:
                correct_preds_batch, = sess.run([accuracy],
                                            feed_dict={
                                                maps: X_batch,
                                                labels: Y_batch
                                            })
            test_acc = 100 * correct_preds_batch / num_test
            print("Test Accuracy {0}".format(test_acc))



def get_next_batch(trn_btch_str, trn_btch_end, epoch_num, num_train):
    trn_btch_str = trn_btch_end - 1
    trn_btch_end = trn_btch_str + cnn_obj.BATCH_SIZE
    if trn_btch_end > num_train:
        trn_btch_str = 0
        trn_btch_end = trn_btch_str + cnn_obj.BATCH_SIZE
        epoch_num += 1
        if epoch_num % 5 == 0:
            epoch_num += 5
    return trn_btch_str, trn_btch_end, epoch_num


if __name__ == "__main__":
    if (len(sys.argv) >1 and sys.argv[1] == "no_congestion"):
        congestion_enabled = 0
        print("CNN training: Congestion Disabled")
    else:
        congestion_enabled = 1 
    train(congestion_enabled)
