#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:LXM
@file:eval.py
@time:2020/10/09
"""

import time
import tensorflow.compat.v1 as tf
import random
import mnist_inference
import mnist_train
import tensorflow.keras as keras
from keras.utils import np_utils


# 加载的时间间隔。
EVAL_INTERVAL_SECS = 10

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
def evaluate(x_test,y_test):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
        validate_feed = {x: x_test, y_: y_test}

        y = mnist_inference.inference(x, None)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
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
    x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1) / 255.0
    # 转one-hot标签
    y_train = np_utils.to_categorical(y_train, num_classes=10)
    y_test = np_utils.to_categorical(y_test, num_classes=10)

    evaluate(x_test,y_test)

if __name__ == '__main__':
    tf.disable_eager_execution()
    main()