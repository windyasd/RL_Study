#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:LXM
@file:DeepRNN_KERAS.py
@time:2020/10/13
"""

import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow import keras

# 定义RNN参数
HIDDEN_SIZE = 30                            # LSTM中隐藏节点的个数。
NUM_LAYERS = 2                              # Deep_LSTM的层数。
TIMESTEPS = 10                              # 循环神经网络的训练序列长度。
TRAINING_STEPS = 10000                      # 训练轮数。
BATCH_SIZE = 32                             # batch大小。
TRAINING_EXAMPLES = 10000                   # 训练数据个数。
TESTING_EXAMPLES = 1000                     # 测试数据个数。
SAMPLE_GAP = 0.01                           # 采样间隔。

# 正弦函数采样
def generate_data(seq):
    X = []
    y = []
    # 序列的第i项和后面的TIMESTEPS-1项合在一起作为输入；第i + TIMESTEPS项作为输
    # 出。即用sin函数前面的TIMESTEPS个点的信息，预测第i + TIMESTEPS个点的函数值。
    for i in range(len(seq) - TIMESTEPS):
        X.append([seq[i: i + TIMESTEPS]])
        y.append([seq[i + TIMESTEPS]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


if __name__ == '__main__':
    tf.disable_eager_execution()
    # 用正弦函数生成训练和测试数据集合。
    test_start = (TRAINING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
    test_end = test_start + (TESTING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
    train_X, train_y = generate_data(np.sin(np.linspace(
        0, test_start, TRAINING_EXAMPLES + TIMESTEPS, dtype=np.float32)))
    test_X, test_y = generate_data(np.sin(np.linspace(
        test_start, test_end, TESTING_EXAMPLES + TIMESTEPS, dtype=np.float32)))

    inputs = keras.Input(shape=(1, TIMESTEPS))
    cell = tf.nn.rnn_cell.MultiRNNCell([
        tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        for _ in range(NUM_LAYERS)])
    output = layers.RNN(cell)(inputs)
    predictions = layers.Dense(1, activation=None)(output)
    model = keras.Model(inputs=inputs, outputs=predictions, name="DeepRNN_model")
    model.summary()
    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adagrad(learning_rate=0.1),
        metrics=["acc"]
    )
    model.fit(train_X, train_y, batch_size=BATCH_SIZE, epochs=10)
    model.evaluate(test_X, test_y)


