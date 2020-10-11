#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:LXM
@file:LeNet5_eval.py
@time:2020/10/11
"""

import time
import tensorflow.compat.v1 as tf
import tensorflow.keras as keras
from keras.utils import np_utils
import numpy as np
# 加载 LeNet5_inference.py 和 LeNet5_train .py中定义的常量和函数。
import LeNet5_inference
import LeNet5_train



# 加载的时间间隔。
EVAL_INTERVAL_SECS = 10

def evaluate(x_test,y_test):
    with tf.Graph().as_default() as g:
        # 定义输入输出的格式
        x = tf.placeholder(tf.float32, [
            None,
            LeNet5_inference.IMAGE_SIZE,
            LeNet5_inference.IMAGE_SIZE,
            LeNet5_inference.NUM_CHANNELS],
                           name='x-input')
        y_ = tf.placeholder(tf.float32, [None, LeNet5_inference.OUTPUT_NODE], name='y-input')

        validate_feed = {x: x_test, y_: y_test}
        # 直接通过调用封装好的函数来计算前向传播的结果，因为测试时不关注正则化损失的值，故设置为NONE
        y = LeNet5_inference.inference(x, None, LeNet5_train.REGULARIZATION_RATE)
        # 使用前向传播的结果计算正确率。如果需要对未知的样例进行分类，那么使用tf.argmax(y_, 1)就可以得到预测的类别
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # 通过变量重命名的方式来加载模型，这样在前向传播的过程中就不需要调用求滑动平均的函数来获取平均值了。这样就可以完全共用
        # LeNet5_inference.py中定义的前向传播过程
        variable_averages = tf.train.ExponentialMovingAverage(LeNet5_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # 每隔EVALINTERVAL_SECS秒调用一次计算正确率的过程以检测训练过程中正确率的＃变化。
        while True:
            with tf.Session() as sess:
                # tf.train.get_checkpoint_state函数会通过checkpoint文件自动找到目录中最新模型的文件名
                ckpt = tf.train.get_checkpoint_state(LeNet5_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 通过文件名得到模型保存时迭代的轮数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    #归一化
    x_train = x_train.reshape(x_train.shape[0], LeNet5_inference.IMAGE_SIZE,LeNet5_inference.IMAGE_SIZE, 1)
    x_test = x_test.reshape(x_test.shape[0], LeNet5_inference.IMAGE_SIZE,LeNet5_inference.IMAGE_SIZE, 1)
    x_train=x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train/=255.0
    x_test /= 255.0
    # 转one-hot标签
    y_train = np_utils.to_categorical(y_train, num_classes=10)
    y_test = np_utils.to_categorical(y_test, num_classes=10)

    evaluate(x_test,y_test)

if __name__ == '__main__':
    tf.disable_eager_execution()
    main()
