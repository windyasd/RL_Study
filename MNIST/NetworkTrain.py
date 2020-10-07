#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:LXM
@file:NetworkTrain.py
@time:2020/10/06

从keras导入MNIST数据集
注意keras中的数据集输出为具体的值，而不是一个向量
"""
import tensorflow.compat.v1 as tf
from keras.utils import np_utils
import tensorflow.keras as keras
import random
import numpy as np


INPUT_NODE = 784  # 输入节点
OUTPUT_NODE = 10  # 输出节点
LAYER1_NODE = 500  # 隐藏层节点数
BATCH_SIZE = 100  # 每次batch打包的样本个数
# 模型相关的参数
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 10000
MOVING_AVERAGE_DECAY = 0.99

def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    """2. 定义辅助函数来计算前向传播结果，使用ReLU做为激活函数"""
    # 不使用滑动平均类
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2

    else:
        # 使用滑动平均类
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


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
    if now_batch < total_batch-1:
        image_batch = image[now_batch*batch_size:(now_batch+1)*batch_size]
        label_batch = label[now_batch*batch_size:(now_batch+1)*batch_size]
    else:
        image_batch = image[now_batch*batch_size:]
        label_batch = label[now_batch*batch_size:]
    return image_batch, label_batch
def train():
    """定义训练过程"""
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    # 生成隐藏层的参数。
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    # 生成输出层的参数。
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算不含滑动平均类的前向传播结果
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 定义训练轮数及相关的滑动平均类
    global_step = tf.Variable(0, trainable=False)           #设定为不可训练的变量
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())   #对参数进行滑动平均
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)   #计算前向传播结果

    # 计算交叉熵及其平均值
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 损失函数的计算
    regularzation = REGULARAZTION_RATE * tf.nn.l2_loss(weights1) + REGULARAZTION_RATE * tf.nn.l2_loss(weights2)
    # regularizer=tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    # #计算模型的正则化损失，一般只计算神经网络边上权重的正则化损失，而不是用偏置项
    # regularaztion = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularzation

    # 设置指数衰减的学习率。
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        60000 / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)

    # 优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 反向传播更新参数和更新每一个参数的滑动平均值
    # 在训练、神经网络模型时，每过一遍数据既需要通过反向传播来更新神经网络中的参数，
    #又要更新每一个参数的滑到J平均值。为了一次完成多个操作， TensorFlow 提供了
    # tf.control dependencies 和 tf.group 两种机制 。 下面两行程序和
    # train_op = tf . group(train_step, variables_averages_op ）是等价的
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 计算正确率

    #检验使用了滑动平均模型的神经网络前向传播结果是否正确。tf.argmax(average_y , 1)
    #计算每一个样例的预测答案。其中 average_y 是一个 batch_size*10 的二维数组，每一行
    #表示一个样例的前向传播结果。 tf.argmax 的第二个参数＂ l ”表示选取最大值的操作仅在第一
    #个维度中进行，也就是说，只在每一行选取最大值对应的下标。于是得到的结果是一个长度为
    #batch 的一维数组，这个一维数组中的值就表示了每一个样例对应的数字识别结果。tf.equal
    #判断两个张量的每一维是否相等，如果相等返回True，否则返回 False
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化会话，并开始训练过程。
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
        x_test = x_test.reshape(x_test.shape[0], -1) / 255.0
        # 转one-hot标签
        y_train = np_utils.to_categorical(y_train, num_classes=10)
        y_test = np_utils.to_categorical(y_test, num_classes=10)
        x_train, y_train, x_test, y_test = shuffle_set(x_train, y_train, x_test, y_test)

        # train_db = tf.data.Dataset.from_tensor_slices({x: x_train, y_: y_train})
        # test_db=tf.data.Dataset.from_tensor_slices({x: x_test, y: y_test})
        # # test_db=test_db.shuffle(1000).batch(512)
        # validate_feed=train_db.shuffle(1000)
        validate_feed = {x: np.reshape(x_train[1:10001],(-1,784)), y_: y_train[1:10001]}

        test_feed = {x: np.reshape(x_test,(-1,784)), y_: y_test}

        # 循环的训练神经网络。
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy using average model is %g " % (i, validate_acc))


            xs, ys = get_batch(x_train,y_train,BATCH_SIZE,i%599,600)
            sess.run(train_op, feed_dict={x: np.reshape(xs,(-1,784)), y_: ys})

        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print(("After %d training step(s), test accuracy using average model is %g" % (TRAINING_STEPS, test_acc)))

def main(argv=None):
    train()

if __name__=='__main__':
    tf.disable_eager_execution()
    main()
