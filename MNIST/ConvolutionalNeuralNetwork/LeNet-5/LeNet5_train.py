#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:LXM
@file:LeNet5_train.py
@time:2020/10/10
"""
import tensorflow.compat.v1 as tf
import LeNet5_inference
from keras.utils import np_utils
import random
import tensorflow.keras as keras
import os
import numpy as np

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 10000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH="C:\SoftwareFilePath\Python_code\RL_Study\MNIST\ConvolutionalNeuralNetwork/"
MODEL_NAME="LeNet5_model"

def shuffle_set(train_image, train_label, test_image, test_label):
    train_row = list(range(len(train_label)))
    random.shuffle(train_row)
    train_image = train_image[train_row]
    train_label = train_label[train_row]

    test_row = list(range(len(test_label)))
    random.shuffle(test_row)
    test_image = test_image[test_row]
    test_label = test_label[test_row]
    return train_image, train_label, test_image, test_label


def get_batch(image, label, batch_size, now_batch, total_batch):
    if now_batch < total_batch - 1:
        image_batch = image[now_batch * batch_size:(now_batch + 1) * batch_size]
        label_batch = label[now_batch * batch_size:(now_batch + 1) * batch_size]
    else:
        image_batch = image[now_batch * batch_size:]
        label_batch = label[now_batch * batch_size:]
    return image_batch, label_batch
def train(x_train, y_train):
    # 定义输出为4维矩阵的placeholder
    x = tf.placeholder(tf.float32, [
        None,
        LeNet5_inference.IMAGE_SIZE,
        LeNet5_inference.IMAGE_SIZE,
        LeNet5_inference.NUM_CHANNELS],
                       name='x-input')
    y_ = tf.placeholder(tf.float32, [None, LeNet5_inference.OUTPUT_NODE], name='y-input')


    y = LeNet5_inference.inference(x, True, REGULARIZATION_RATE)
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数、学习率、滑动平均操作以及训练过程。
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        len(y_train) / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 初始化TensorFlow持久化类。
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            Iteration_data = int(len(y_train) / BATCH_SIZE)  # 自己编写的getBatch函数,确定batchs的个数
            xs, ys = get_batch(x_train, y_train, BATCH_SIZE, i % Iteration_data, Iteration_data)


            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})

            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def main(argv=None):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # 将输出处理为相应的格式
    x_train = x_train.reshape(x_train.shape[0], LeNet5_inference.IMAGE_SIZE, LeNet5_inference.IMAGE_SIZE, 1)
    x_test = x_test.reshape(x_test.shape[0], LeNet5_inference.IMAGE_SIZE, LeNet5_inference.IMAGE_SIZE, 1)
    # 归一化
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0
    # 转one-hot标签
    y_train = np_utils.to_categorical(y_train, num_classes=10)
    y_test = np_utils.to_categorical(y_test, num_classes=10)
    # 打乱数据集的顺序
    x_train, y_train, x_test, y_test = shuffle_set(x_train, y_train, x_test, y_test)
    tf.disable_eager_execution()
    train(x_train, y_train)

if __name__ == '__main__':
    main()