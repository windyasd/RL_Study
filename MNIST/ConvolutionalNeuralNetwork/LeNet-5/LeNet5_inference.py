#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:LXM
@file:mnist_inference.py
@time:2020/10/10
"""
import tensorflow.compat.v1 as tf

# 配置神经网络的参数
INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10
# 第一层卷积层的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 5
# 第二层卷积层的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 5
#全连接层的节点个数
FC_SIZE = 512

def inference(input_tensor, train, REGULARIZATION_RATE):
    """
    通过使用不同的命名空间来隔离不同层的变量,这可以让每一层的变量命名只需要考虑在当前层的作用,无需担心重名的问题
    :param input_tensor:
    :param train:
    :param REGULARIZATION_RATE:
    :return:
    """
    # 声明第一层卷积层的变量，定义的卷积层输入为 28x28xl 的原始MNIST图片像素。因为卷积层中使用了全0填充，所以输出为28*28*32的矩阵
    # 使用size为5*5，deepth为32的过滤器，移动步长为1
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(
            "weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # 实现第二层池化层的前向传播过程。选用最大池化层，size为2，全零填充，步长为2，输出为14*14*32
    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")

    # 实现第三层，步长为1，输入为14*14*32，输出为14*14*64
    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable(
            "weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # 实现第四层池化层的前向传播过程，输入为14*14*64，输出为7*7*64。
    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # pool_shape[0] 为输入的batch大小
        pool_shape = pool2.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.layers.flatten(pool2)     # 第四层的输出转化为第五层全连接层的输入，要将其拉直成一个向量

    # 声明第五层全连接层的变量并实现前向传播过程。这里引入了dropout，在训练时会随机将部分节点的输出改为0，可以避免过拟合问题
    # dropout一般只在全连接层而不是卷积层或池化层使用
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if REGULARIZATION_RATE != None:
            tf.add_to_collection('losses', REGULARIZATION_RATE * tf.nn.l2_loss(fc1_weights))
        fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    # 声明第六层全连接层的变量并实现前向传播过程，输入为512的向量，输出为一组长度为10的向量。这一层的输出通过softmax后就得到了最后
    # 的分类结果
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable("weight", [FC_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if REGULARIZATION_RATE != None:
            tf.add_to_collection('losses', REGULARIZATION_RATE * tf.nn.l2_loss(fc2_weights))
        fc2_biases = tf.get_variable("bias", [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit