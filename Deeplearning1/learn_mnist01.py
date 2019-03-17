#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow.examples.tutorials.mnist.input_data as input_data
if __name__=="__main__":

    import tensorflow as tf
    import tensorflow.examples.tutorials.mnist.input_data as input_data

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    print(mnist.train.images.shape)
    print(mnist.train.labels.shape)
    print(mnist.validation.images.shape)
    print(mnist.validation.labels.shape)
    print(mnist.test.images.shape)
    print(mnist.test.labels.shape)
    x = tf.placeholder("float", [None, 784])  # 输入图像
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder("float", [None, 10])
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))  # 使用交叉熵
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)  # 使用梯度下降
    init = tf.global_variables_initializer()  # 全局变量初始化
    sess = tf.Session()  # 开启图
    sess.run(init)
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)  # 图像和标签，该循环中每步选取100个数据
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        print(">" * (i // 10), i)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  # 进行评估
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # 转换数据类型
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

