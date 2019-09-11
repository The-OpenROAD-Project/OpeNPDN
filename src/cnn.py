#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 18:44:26 2019
This script defines the complete CNN, with the layers, forward propagation and
training. The various hyper paramters are defined.
@author: chhab011
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import cnn_input
import math
# Hyper parameter definition
NUM_NEURONS = 1024
LEARNING_RATE = 0.01
NUM_EPOCHS_PER_DECAY = 5
EPOCH_SIZE = 32
LEARNING_RATE_DECAY_FACTOR = 0.1
BATCH_SIZE = 64

MOMEMTUM = 0.1
DROPOUT = 0.75


def inference(X,cong):
    """ Performes the forward propagation in the CNN during training
    Args:
        X: This is the input to the CNN, current maps in our case
        cong: This is the input to the CNN, congestion in our case
    Returns:
        logits: Output class after the softmax layer. This is the output of the
                last layer of the CNN
    """

    # First layer definition: Convolution layer
    with tf.variable_scope('conv1') as scope:
        images_in = tf.reshape(X,
                            shape=[
                                -1, cnn_input.MAP_SIZE_X,
                                cnn_input.MAP_SIZE_Y, 1
                            ])
        cong_in =  tf.reshape(cong,
                            shape=[
                                -1, cnn_input.MAP_SIZE_X,
                                cnn_input.MAP_SIZE_Y, 1
                            ])
        images = tf.concat([images_in,cong_in],3)
        kernels = tf.get_variable('kernels',
                                  shape=([5, 5, 2, 32]),
                                  initializer=tf.truncated_normal_initializer())
        biases = tf.get_variable('biases',
                                 shape=([32]),
                                 initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(images, kernels, [1, 1, 1, 1], "SAME")
        conv1_norm = tf.layers.batch_normalization(conv)
        conv1 = tf.nn.relu(conv1_norm + biases, name=scope.name)

    # Second layer definition: Max pooling layer
    with tf.variable_scope('pool1') as scope:
        pool1 = tf.nn.max_pool(conv1,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
    
    # Third layer definition: Convolution layer
    with tf.variable_scope('conv2') as scope:
        kernels = tf.get_variable('kernels',
                                  shape=([3, 3, 32, 64]),
                                  initializer=tf.truncated_normal_initializer())
        biases = tf.get_variable('biases',
                                 shape=([64]),
                                 initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(pool1, kernels, [1, 1, 1, 1], "SAME")
        conv2_norm = tf.layers.batch_normalization(conv)
        conv2 = tf.nn.relu(conv2_norm + biases, name=scope.name)
    
    # Fourth layer definition: Max pooling layer
    with tf.variable_scope('pool2') as scope:
        pool2 = tf.nn.max_pool(conv2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
    
    # Fifth layer definition: Fully connected layer
    with tf.variable_scope('fc') as scope:
        fc_in = pool2
        input_features = int(math.ceil(math.ceil(cnn_input.MAP_SIZE_X/2)/2) *
                             math.ceil(math.ceil(cnn_input.MAP_SIZE_Y/2)/2) * 64)
        fc_in1 = tf.reshape(fc_in, [-1, input_features])
        #fc_w_c = tf.concat([fc_in1,cong],1)
        w = tf.Variable(tf.truncated_normal([input_features, NUM_NEURONS]),
                        name='fc1_w')
        b = tf.Variable(tf.truncated_normal([NUM_NEURONS]), name='fc1_b')
        fc = tf.nn.relu(tf.matmul(fc_in1, w) + b, name='relu')
        #fc1 = tf.nn.dropout(fc,DROPOUT, name = 'drop_out_relu')

    # Sixth layer definition: Softmax layer for classification
    with tf.variable_scope('softmax_linear') as scope:
        w = tf.Variable(tf.zeros([NUM_NEURONS, cnn_input.N_CLASSES]))
        b = tf.Variable(tf.zeros([cnn_input.N_CLASSES]))
        logits = tf.matmul(fc, w) + b
    return logits


def loss(logits, labels):
    """ This function calculates the cross entropy loss given the output of the
    CNN and the actual correct labels
    Args:
        logits: This is the output of the CNN during the forward propagation 
        labels: This is the actal class to wich the data belongs to
    Returns:
        total_loss: Total cross entropy loss for a given label
    """
    ent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(ent, 0)
    #provison to add l2 and l1 losses
    total_loss = loss
    return total_loss


def train(total_loss, global_step):
    """ This functions trains the weights of the CNN by trying to minimize loss
    and with a fixed step size
    Args:
        total_loss: This is the output of the CNN during the forward propagation 
        global_step: This is the actual class to which the data belongs to
    Returns:
        optimizer: Th optimized model with the weights after minimizing
        total_loss 
    """
    num_batches_epoch = EPOCH_SIZE / BATCH_SIZE
    decay_steps = int(num_batches_epoch * NUM_EPOCHS_PER_DECAY)
    lr = tf.train.exponential_decay(LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(lr).minimize(total_loss,global_step = global_step)
    return optimizer


def accuracy(logits, labels):
    """ This function finds the accuracy of the traning data classification
    using the trained and optimized CNN
    Args:
        logits: the output produced by the optimized CNN
        labels: the actual output produced by the golden algorithm
    Returns:
        accuracy: Number of inputs correctly classified by the total number of
        inputs
    """
    preds = tf.nn.softmax(logits)
    correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
    return accuracy

def prediction(logits):
    """ This function finds the prediction of the traning data classification
    using the trained and optimized CNN
    Args:
        logits: the output produced by the optimized CNN
    Returns:
        predict: Predicted ouput template for the corresponding inputs
    """
    preds = tf.nn.softmax(logits)
    predict = tf.argmax(preds, 1)
    return predict
