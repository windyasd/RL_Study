#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:LXM
@file:mnist_inference.py
@time:2020/10/09
"""

import tensorflow.compat.v1 as tf

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500


def get_weight_variable(shape, REGULARAZTION_RATE):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if REGULARAZTION_RATE is not None:
        tf.add_to_collection('losses', REGULARAZTION_RATE * tf.nn.l2_loss(weights))
    return weights


def inference(input_tensor, REGULARAZTION_RATE):
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], REGULARAZTION_RATE)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], REGULARAZTION_RATE)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2
