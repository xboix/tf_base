"""
The model is adapted from the tensorflow tutorial:
https://www.tensorflow.org/get_started/mnist/pros
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from networks import network
from tensorflow.python.training import py_checkpoint_reader


class CNN(network.Network):

    def __init__(self, config, num_features):
        super().__init__(config['num_classes'])

        initial_learning_rate = float(config['initial_learning_rate'])
        training_batch_size = config['training_batch_size']
        batch_decrease_learning_rate = float(config['batch_decrease_learning_rate'])

        self.num_features = num_features
        self.batch_size = training_batch_size

        self.mode = 'train'

        self.train_variables = []
        if num_features == 32*32*3: #CIFAR
            self.num_channels = 3
            self.feat_size = 8
            self.im_size = 32
        else: #MNIST
            self.num_channels = 1
            self.feat_size = 7
            self.im_size = 28

        # first convolutional layer
        self.W_conv1 = self._weight_variable([5, 5, self.num_channels, 32], name="Variable")
        self.train_variables += [self.W_conv1]
        self.b_conv1 = self._bias_variable([32], name="Variable_1")
        self.train_variables += [self.b_conv1]

        # second convolutional layer
        self.W_conv2 = self._weight_variable([5, 5, 32, 64], name="Variable_2")
        self.train_variables += [self.W_conv2]
        self.b_conv2 = self._bias_variable([64], name="Variable_3")
        self.train_variables += [self.b_conv2]

        # first fully connected layer
        self.W_fc1 = self._weight_variable([self.feat_size * self.feat_size * 64, 1024], name="Variable_4")
        self.train_variables += [self.W_fc1]
        self.b_fc1 = self._bias_variable([1024], name="Variable_5")
        self.train_variables += [self.b_fc1]

        # output layer
        self.W_fc2 = self._weight_variable([1024, 10], name="Variable_6")
        self.train_variables += [self.W_fc2]
        self.b_fc2 = self._bias_variable([10], name="Variable_7")
        self.train_variables += [self.b_fc2]

        # Setting up the optimizer
        self.learning_rate = \
            tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,
                                                           training_batch_size * batch_decrease_learning_rate, 0.85,
                                                           staircase=True)

        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

    def feedforward_pass(self, input):

        self.x_input_image = tf.reshape(input, [-1, self.im_size, self.im_size, self.num_channels])

        self.z1 = self._conv2d(self.x_input_image, self.W_conv1) + self.b_conv1
        h_conv1 = tf.nn.relu(self.z1)
        self.h_pool1 = self._max_pool_2x2(h_conv1)
        if self.mode == "train":
            self.indices_1 = tf.reshape(tf.vectorized_map(self.pool_argmax, h_conv1), tf.shape(self.h_pool1))

        h_conv2 = tf.nn.relu(self._conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
        self.h_pool2 = self._max_pool_2x2(h_conv2)
        if self.mode == "train":
            self.indices_2 = tf.reshape(tf.vectorized_map(self.pool_argmax, h_conv2), tf.shape(self.h_pool2))

        h_pool2_flat = tf.reshape(self.h_pool2, [-1, self.feat_size * self.feat_size * 64])
        self.h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.W_fc1) + self.b_fc1)

        self.pre_softmax = tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2
        return self.pre_softmax

    def set_mode(self, mode='train'):
        self.mode = mode

    @staticmethod
    def _weight_variable(shape, name):
        initial = tf.keras.initializers.GlorotUniform()
        return tf.Variable(initial_value=initial(shape), name=name, trainable=True)

    @staticmethod
    def _bias_variable(shape, name):
        initial = tf.constant(0.1, shape=shape, name=name)
        return tf.Variable(initial)

    @staticmethod
    def _conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def _max_pool_2x2(x):
        return tf.nn.max_pool(x,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

    @tf.function
    def pool_argmax(self, args):
        return tf.nn.max_pool_with_argmax(args[None], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                          include_batch_in_index=True)[1]

    @tf.function
    def _gather_max(self, args):
        x, indices = args
        return tf.gather(tf.reshape(x, [-1]), indices)
