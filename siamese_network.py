from __future__ import print_function
from __future__ import division

from pdb import set_trace as debugger
from tensorflow.python.training import moving_averages

import tensorflow as tf
import numpy as np

import os
import sys

#batch_size = 10
xsize, ysize = 50, 50
resnet_units = 3

class ModeSiameseNetwork(object):
    def __init__(self):

        tf.reset_default_graph()

        # Create model
        self.x1 = tf.placeholder(tf.float32, [None, xsize* ysize])
        self.x2 = tf.placeholder(tf.float32, [None, xsize* ysize])
        x1_r = tf.reshape(self.x1, [-1, xsize, ysize])
        x2_r = tf.reshape(self.x2, [-1, xsize, ysize])
        self.is_training = tf.placeholder(tf.bool)

        with tf.variable_scope("siamese") as scope:
            siamese1 = self.residual_network(x1_r)
            scope.reuse_variables()
            siamese2 = self.residual_network(x2_r)

        # Calculate energy, loss, and accuracy
        self.margin = tf.placeholder(tf.float32)
        self.y_ = tf.placeholder(tf.float32, [None, 1])
        self.energy_op = self.compute_energy(siamese1, siamese2)
        self.loss_op = self.compute_loss(self.y_, self.energy_op)
        self.accuracy_op = self.compute_accuracy(self.y_, self.energy_op)

        # setup siamese network
        self.train_step = tf.train.AdamOptimizer(1e-4, beta1=0.5).minimize(self.loss_op)

       


    def global_avg_pool(self, in_var, name='global_pool'):
        assert name is not None, 'Op name should be specified'
        # start global average pooling
        with tf.name_scope(name):
            input_shape = in_var.get_shape()
            assert len(input_shape) == 4, 'Incoming Tensor shape must be 4-D'

            inference = tf.reduce_mean(in_var, [1, 2])
            return inference


    def max_pool(self, in_var, kernel_size=[1,2,2,1], strides=[1,1,1,1], 
                padding='SAME', name=None):
        assert name is not None, 'Op name should be specified'
        assert strides is not None, 'Strides should be specified when performing max pooling'
        # start max pooling
        with tf.name_scope(name):
            input_shape = in_var.get_shape()
            assert len(input_shape) == 4, 'Incoming Tensor shape must be 4-D'

            inference = tf.nn.max_pool(in_var, kernel_size, strides, padding)
            return inference


    def avg_pool_2d(self, in_var, kernel_size=[1,2,2,1], strides=None, 
                    padding='SAME', name=None):
        assert name is not None, 'Op name should be specified'
        assert strides is not None, 'Strides should be specified when performing average pooling'
        # start average pooling
        with tf.name_scope(name):
            input_shape = in_var.get_shape()
            assert len(input_shape) == 4, 'Incoming Tensor shape must be 4-D'

            inference = tf.nn.avg_pool(in_var, kernel_size, strides, padding)
            return inference

    def conv_2d(self, in_var, out_channels, filters=[3,3], strides=[1,1,1,1], 
                padding='SAME', name=None):
        assert name is not None, 'Op name should be specified'
        # start conv_2d
        with tf.name_scope(name):
            k_w, k_h = filters  # filter width/height
            W = tf.get_variable(name + "_W", [k_h, k_w, in_var.get_shape()[-1], out_channels],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
            b = tf.get_variable(name + "_b", [out_channels], initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv2d(in_var, W, strides=strides, padding=padding)
            #conv = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())
            conv = tf.nn.bias_add(conv, b)

            return conv

    def residual_block(self, in_var, nb_blocks, out_channels, batch_norm=True, strides=[1,1,1,1],
                        downsample=False, downsample_strides=[1,2,2,1], name=None):
        assert name is not None, 'Op name should be specified'
        # start residual block
        with tf.name_scope(name):
            resnet = in_var
            in_channels = in_var.get_shape()[-1].value

            # multiple layers for a single residual block
            for i in xrange(nb_blocks):
                identity = resnet
                ##################
                # apply convolution
                resnet = self.conv_2d(resnet, out_channels, 
                                strides=strides if not downsample else downsample_strides, 
                                name='{}_conv2d_{}'.format(name, i))
                # normalize batch before activations
                if batch_norm:
                    resnet = self.batch_normalization(resnet, name='{}_batch_norm{}'.format(name, i))
                # apply activation function
                resnet = tf.nn.relu(resnet)
                # apply convolution again
                resnet = self.conv_2d(resnet, out_channels, strides=strides, name='{}_conv2d_{}{}'.format(name, i, 05))
                # normalize batch before activations or previous convolution
                if batch_norm:
                    resnet = self.batch_normalization(resnet, name='{}_batch_norm{}'.format(name, i), reuse=True)
                # apply activation function
                resnet = tf.nn.relu(resnet)
                ##################
                # downsample
                if downsample:
                    identity = self.avg_pool_2d(identity, strides=downsample_strides, name=name+'_avg_pool_2d')
                # projection to new dimension by padding
                if in_channels != out_channels:
                    ch = (out_channels - in_channels)//2
                    identity = tf.pad(identity, [[0, 0], [0, 0], [0, 0], [ch, ch]])
                    in_channels = out_channels
                # add residual
                resnet = resnet + identity

            return resnet

    def batch_normalization(self, in_var, beta=0.0, gamma=1.0, epsilon=1e-5, 
                            decay=0.9, name=None, reuse=None):
        assert name is not None, 'Op name should be specified'
        # start batch normalization with moving averages
        input_shape = in_var.get_shape().as_list()
        input_ndim = len(input_shape)

        with tf.variable_scope(name, reuse=reuse):
            gamma_init = tf.random_normal_initializer(mean=1.0, stddev=0.002)
            beta = tf.get_variable(name+'_beta', shape=[input_shape[-1]],
                                   initializer=tf.constant_initializer(beta),
                                   trainable=True)
            gamma = tf.get_variable(name+'_gamma', shape=[input_shape[-1]],
                                    initializer=gamma_init, trainable=True)

            axis = list(range(input_ndim - 1))
            moving_mean = tf.get_variable(name+'_moving_mean',
                            input_shape[-1:], initializer=tf.zeros_initializer,
                            trainable=False)
            moving_variance = tf.get_variable(name+'_moving_variance',
                                input_shape[-1:], initializer=tf.ones_initializer,
                                trainable=False)

            # define a function to update mean and variance
            def update_mean_var():
                mean, variance = tf.nn.moments(in_var, axis)
                update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, decay)
                update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, decay)
                with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                    return tf.identity(mean), tf.identity(variance)

            # only update mean and variance with moving average while training
            mean, var = tf.cond(self.is_training, update_mean_var, lambda: (moving_mean, moving_variance))

            inference = tf.nn.batch_normalization(in_var, mean, var, beta, gamma, epsilon)
            inference.set_shape(input_shape)

        return inference


    def residual_network(self, x):
        with tf.name_scope('residual_network') as scope:
            _x = tf.reshape(x, [-1, xsize, ysize, 1])
            net = self.conv_2d(_x, 8, filters=[7,7], strides=[1,2,2,1], name='conv_0')
            net = self.max_pool(net, name='max_pool_0')
            net = self.residual_block(net, resnet_units, 8, name='resblock_1')
            net = self.residual_block(net, 1, 16, downsample=True, name='resblock_1-5')
            net = self.residual_block(net, resnet_units, 16, name='resblock_2')
            net = self.residual_block(net, 1, 24, downsample=True, name='resblock_2-5')
            net = self.residual_block(net, resnet_units+1, 24, name='resblock_3')
            net = self.residual_block(net, 1, 32, downsample=True, name='resblock_3-5')
            net = self.residual_block(net, resnet_units, 32, name='resblock_4')
            net = self.batch_normalization(net, name='batch_norm')
            net = tf.nn.relu(net)
            net = self.global_avg_pool(net)
            return net

    def compute_energy(self, o1, o2):
        with tf.name_scope('energy'):
            _energy = tf.reduce_sum(tf.abs(tf.subtract(o1, o2)), reduction_indices=[1], keep_dims=True)
            return _energy

    def compute_loss(self, y_, energy):
        with tf.name_scope('loss'):
            labels_t = y_
            labels_f = tf.subtract(1.0, y_, name="1-y")
            # compute loss_g and loss_i
            loss_g = tf.multiply(tf.truediv(2.0, self.margin), tf.pow(energy, 2), name='l_G')
            loss_i = tf.multiply(tf.multiply(2.0, self.margin), tf.exp(tf.multiply(tf.truediv(-2.77, self.margin), energy)), name='l_I')
            # compute full loss
            pos = tf.multiply(labels_t, loss_g, name='1-Yl_G')
            neg = tf.multiply(labels_f, loss_i, name='Yl_I')
            _loss = tf.reduce_mean(tf.add(pos, neg), name='loss')
            return _loss

    def compute_accuracy(self, true_y, pred_y):
        with tf.name_scope('accuracy'):
            _pred_y = tf.cast(tf.less(pred_y, self.margin/2), tf.float32)
            _acc = tf.reduce_mean(tf.cast(tf.equal(true_y, _pred_y), tf.float32))
            return _acc


        