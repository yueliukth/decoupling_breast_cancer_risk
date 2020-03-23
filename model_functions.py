import os
import time
import math
import numpy as np
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import random
from exp import *

class ResNetGN(object):
  def __init__(self, depth=50, gn_epsilon=1e-5):    
    self._depth = depth
    self._group = 32
    self._gn_epsilon = gn_epsilon
    self._conv_initializer = tf.glorot_uniform_initializer
    self._conv_bn_initializer = tf.glorot_uniform_initializer
    self._block_settings = {
      50: (3, 4, 6, 3),
      101: (3, 4, 23, 3)
    }

  def get_model(self, inputs, ages, training):
    with tf.variable_scope('resnet'):
      input_depth = [128, 256, 512, 1024]
      num_units = self._block_settings[self._depth]
      
      with tf.variable_scope('block_0'):
        inputs = tf.pad(inputs, paddings = [[0, 0], [3, 3], [3, 3], [0, 0]], name='padding_conv1')
        inputs = self.conv_gn_relu(inputs, input_depth[0] // 2, (7, 7), (2, 2), 'conv_1', training, reuse=None, padding='valid')
        inputs = tf.layers.max_pooling2d(inputs, [3, 3], [2, 2], padding='valid', name='pool_1')
    
      is_root = True
      dictionary = {}
      for ind, num_unit in enumerate(num_units):
        with tf.variable_scope('block_{}'.format(ind+1)):
          need_reduce = True
          for unit_index in range(1, num_unit+1):
            inputs = self.bottleneck_block(inputs, input_depth[ind], 'conv_{}'.format(unit_index), training, need_reduce=need_reduce, is_root=is_root)
            need_reduce = False
            is_root = False
        dictionary['block_{}'.format(ind+1)] = inputs
    
      with tf.variable_scope('logits'):
        inputs = tf.reduce_mean(inputs, [1, 2], name='avg_pooling', keepdims=True)
       
        if 'dropout' in exp_name:
          inputs = tf.layers.dropout(inputs, rate=0.5, training=training, name='dropout_1')
        if 'sigmoid' not in exp_name:
          inputs = tf.layers.dense(inputs, 2)
          inputs = tf.reshape(inputs, [-1, 1, 2])
        else:
          inputs = tf.layers.dense(inputs, 1)
          inputs = tf.reshape(inputs, [-1, 1])
     
    return inputs, ages, dictionary

  def conv_gn_relu(self, inputs, filters, kernel_size, strides, scope, training, padding='same', reuse=None):
    with tf.variable_scope(scope):
      inputs = tf.layers.conv2d(inputs, filters, kernel_size, strides=strides, name='conv2d', use_bias=False, padding=padding, activation=None, kernel_initializer=self._conv_initializer(), bias_initializer=None, reuse=reuse)
      
      inputs = self.group_normalization(inputs, training, self._group, scope='gn')
      if 'leaky' in exp_name:
        inputs = tf.nn.leaky_relu(inputs)  
      elif 'swish' in exp_name:
        inputs = inputs * tf.nn.sigmoid(inputs)
      else:
        inputs = tf.nn.relu(inputs)

    return inputs

  def group_normalization(self, inputs, training, group, scope=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      input_shape = tf.shape(inputs)
      _, H, W, C = inputs.get_shape().as_list()
      gamma = tf.get_variable('scale', shape=[C], dtype=tf.float32, initializer=tf.ones_initializer(), trainable=training)
      beta = tf.get_variable('bias', shape=[C], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=training)
      inputs = tf.reshape(inputs, [-1, input_shape[1], input_shape[2], group, C // group], name='unpack')
      mean, var = tf.nn.moments(inputs, [1, 2, 4], keep_dims=True)
      inputs = (inputs - mean) / tf.sqrt(var + self._gn_epsilon)
      inputs = tf.reshape(inputs, input_shape, name='pack')
      gamma = tf.reshape(gamma, [1, 1, 1, C], name='reshape_gamma')
      beta = tf.reshape(beta, [1, 1, 1, C], name='reshape_beta')
      inputs = inputs * gamma + beta
    return inputs  
    
  def bottleneck_block(self, inputs, filters, scope, training, need_reduce=True, is_root=False, reuse=None):
    with tf.variable_scope(scope):
      strides = 1 if (not need_reduce) or is_root else 2
      shortcut = self.conv_gn(inputs, filters * 2, (1, 1), (strides, strides), 'shortcut', training, padding='valid', reuse=reuse) if need_reduce else inputs
      
      inputs = self.conv_gn_relu(inputs, filters // 2, (1, 1), (1, 1), 'reduce', training, reuse=reuse)
      inputs = tf.pad(inputs, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]], name='padding_conv_3x3')
      inputs = self.conv_gn_relu(inputs, filters // 2, (3, 3), (strides, strides), 'block_3x3', training, padding='valid', reuse=reuse)
      inputs = self.conv_gn(inputs, filters * 2, (1, 1), (1, 1), 'increase', training, reuse=reuse)
      if 'leaky' in exp_name:
        inputs = tf.nn.leaky_relu(inputs + shortcut) 
      elif 'swish' in exp_name:
        inputs = inputs * tf.nn.sigmoid(inputs)
      else:
        inputs = tf.nn.relu(inputs + shortcut)
    return inputs

  def conv_gn(self, inputs, filters, kernel_size, strides, scope, training, padding='same', reuse=None):
    with tf.variable_scope(scope):
      inputs = tf.layers.conv2d(inputs, filters, kernel_size, strides=strides, name='conv2d', use_bias=False, padding=padding, activation=None, kernel_initializer=self._conv_initializer(), bias_initializer=None, reuse=reuse)      
      inputs = self.group_normalization(inputs, training, self._group, scope='gn')
    return inputs 

