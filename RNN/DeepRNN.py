#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:LXM
@file:DeepRNN.py
@time:2020/10/13
"""
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import tensorflow.keras as keras

# 定义RNN参数
HIDDEN_SIZE = 30                            # LSTM中隐藏节点的个数。
NUM_LAYERS = 2                              # Deep_LSTM的层数。
TIMESTEPS = 10                              # 循环神经网络的训练序列长度。
TRAINING_STEPS = 3000                      # 训练轮数。
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



# 定义网络结果和优化步骤
def lstm_model(X, y, is_training):
    # # 定义LSTM结构
    # lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
    # # 使用多层的LSTM结构。使用DropoutWrapper类实现dropOut功能
    # cell = tf.nn.rnn_cell.MultiRNNCell([
    #     tf.nn.rnn_cell.DropoutWrapper(lstm_cell)
    #     for _ in range(NUM_LAYERS)])
    # # 使用多层的LSTM结构。
    cell = tf.nn.rnn_cell.MultiRNNCell([
        tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        for _ in range(NUM_LAYERS)])

    # 使用TensorFlow接口将多层的LSTM结构连接成RNN网络并计算其前向传播结果。dynamic_rnn被keras.layers.RNN代替
    # outputs, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    # output = outputs[:, -1, :]
    output= keras.layers.RNN(cell)(X)

    # 对LSTM网络的输出再做加一层全链接层并计算损失。注意这里默认的损失为平均
    # 平方差损失函数。
    predictions=tf.layers.Dense(1,activation=None)(output)
    # predictions=keras.layers.Dense(1)(output)
    # predictions =tf.contrib.layers.fully_connected(output, 1, activation_fn=None)
    # predictions = tf.contrib.layers.fully_connected(
    #     output, 1, activation_fn=None)

    # 只在训练时计算损失函数和优化步骤。测试时直接返回预测结果。
    if not is_training:
        return predictions, None, None

    # 计算损失函数。
    loss = tf.losses.mean_squared_error(labels=y, predictions=predictions)

    # 创建模型优化器并得到优化步骤。
    # 优化损失函数
    train_op=tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
    # train_op = tf.contrib.layers.optimize_loss(
    #     loss, tf.train.get_global_step(),
    #     optimizer="Adagrad", learning_rate=0.1)


    return predictions, loss, train_op

#
def train(sess, train_X, train_y):
    ds = tf.data.Dataset.from_tensor_slices((train_X, train_y))
    ds = ds.repeat().shuffle(1000).batch(BATCH_SIZE)
    X, y = ds.make_one_shot_iterator().get_next()

    # 定义模型，得到预测结果、损失函数，和训练操作。
    with tf.variable_scope("model"):
        predictions, loss, train_op = lstm_model(X, y, True)
    # 训练模型。
    sess.run(tf.global_variables_initializer())
    for i in range(TRAINING_STEPS):
            _, l = sess.run([train_op, loss])
            if i % 1000 == 0:
                print("train step: " + str(i) + ", loss: " + str(l))


# 定义测试步骤
def run_eval(sess, test_X, test_y):
    # 将测试数据以数据集的方式提供给计算图。
    ds = tf.data.Dataset.from_tensor_slices((test_X, test_y))
    ds = ds.batch(1)
    X, y = ds.make_one_shot_iterator().get_next()

    # 调用模型得到计算结果。这里不需要输入真实的y值。
    with tf.variable_scope("model", reuse=True):
        prediction, _, _ = lstm_model(X, [0.0], False)

    # 将预测结果存入一个数组。
    predictions = []
    labels = []
    for i in range(TESTING_EXAMPLES):
        p, l = sess.run([prediction, y])
        predictions.append(p)
        labels.append(l)

    # 计算rmse作为评价指标。
    predictions = np.array(predictions).squeeze()
    labels = np.array(labels).squeeze()
    rmse = np.sqrt(((predictions - labels) ** 2).mean(axis=0))
    print("Root Mean Square Error is: %f" % rmse)

    # 对预测的sin函数曲线进行绘图。
    plt.figure()
    plt.plot(predictions, label='predictions')
    plt.plot(labels, label='real_sin')
    plt.legend()
    plt.show()

# 执行训练和测试
# 将训练数据以数据集的方式提供给计算图。
if __name__ == '__main__':
    # 用正弦函数生成训练和测试数据集合。
    test_start = (TRAINING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
    test_end = test_start + (TESTING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
    train_X, train_y = generate_data(np.sin(np.linspace(
        0, test_start, TRAINING_EXAMPLES + TIMESTEPS, dtype=np.float32)))
    test_X, test_y = generate_data(np.sin(np.linspace(
        test_start, test_end, TESTING_EXAMPLES + TIMESTEPS, dtype=np.float32)))

    with tf.Session() as sess:
        # 训练模型

        train(sess,train_X,train_y)

        # # 使用训练好的模型对测试数据进行预测。
        # print
        # "Evaluate model after training."
        run_eval(sess, test_X, test_y)
