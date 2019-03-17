#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

def weight_variable(shape):
    """权重初始化"""
    init=tf.random.truncated_normal(shape,stddev=0.1)#使权重为正太分布，偏差为0.1
    return  tf.Variable(init)

def bias_variable(shape):
    """偏置项初始化"""
    init=tf.constant(0.1,shape=shape)
    return  tf.Variable(init)

#这里的4D输入  可以这样来理解：图像的数量，图像行，图像列，图像通道
#filter:[filter_height, filter_width, in_channels, out_channels]`
#这里的w就是filter,输入w是tensor会使用其shape 的值作为输入。
def conv2d(x,w):
    return  tf.nn.conv2d(x,w,[1,1,1,1],'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

model_layers=[]

if __name__=="__main__":
    W_conv1 = weight_variable([5, 5, 1, 32])#卷积核为5x5的窗口,1个通道，32就是指卷积核的数量
    b_conv1 = bias_variable([32])#32个输出通道的bias
    x = tf.placeholder("float", shape=[None, 784])
    x_image = tf.reshape(x, [-1, 28, 28, 1])  # 处理输入的图片，变成需要的形式，
    # -1好像表示可以自动根据其他的几个参数自己变化，
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    print ("h_conv1:",h_conv1.shape)#？[-1, 28, 28, 32]
    h_pool1 = max_pool_2x2(h_conv1)
    print("h_pool1:", h_pool1.shape)#?[-1,14,14,32]

#第二层
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    print("h_conv2:", h_conv2.shape)#?[-1,14,14,64]
    h_pool2 = max_pool_2x2(h_conv2)
    print("h_pool2:", h_pool2.shape)#?[-1,7,7,64]

#全连接层
    # 图片减小到7x7，加入有1024个神经元的全连接层，处理整个图片
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    #乘上权重矩阵，加上偏置，然后对其使用ReLU。
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#DropOut层   减少过拟合的作用
    pass
