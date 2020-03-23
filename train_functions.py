import os
import time
import math
import numpy as np
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import random
from exp import *
from model_functions import *

def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def get_tensors_in_checkpoint_file(file_name,all_tensors=True,tensor_name=None):
  varlist=[]
  var_value =[]
  reader = pywrap_tensorflow.NewCheckpointReader(file_name)
  if all_tensors:
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in sorted(var_to_shape_map):
      varlist.append(key)
      var_value.append(reader.get_tensor(key))
  else:
    varlist.append(tensor_name)
    var_value.append(reader.get_tensor(tensor_name))
  return (varlist, var_value)

def build_tensors_in_checkpoint_file(loaded_tensors):
  full_var_list = list()
  # Loop all loaded tensors
  for i, tensor_name in enumerate(loaded_tensors[0]):
    # Extract tensor
    try:
      tensor_aux = tf.get_default_graph().get_tensor_by_name(tensor_name+":0")
    except:
       print('Not found: '+tensor_name)
    full_var_list.append(tensor_aux)
  return full_var_list

def define_lr_opt(global_step, steps_per_epoch, optimizer, learning_rate_scratch, decay_epoch, num_epochs):
  # define learning rate and optimizer
  if optimizer == 'adam':
    learning_rate = learning_rate_scratch
    opt = tf.train.AdamOptimizer(learning_rate)
    
  elif optimizer == 'sgd':
    learning_rate = tf.train.piecewise_constant(global_step, [steps_per_epoch*decay_epoch] , [learning_rate_scratch, learning_rate_scratch/10]) 
    opt = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    if 'asgd' in exp_name:
      opt = tf.contrib.opt.MovingAverageOptimizer(opt)
  elif optimizer == 'cyclicallr':
    base_lr = 1e-10
#    base_lr = 1e-5
    max_lr = 1e0
    max_base_ratio = max_lr / base_lr
    learning_rate = tf.train.exponential_decay(base_lr, global_step, num_epochs * steps_per_epoch, max_base_ratio)
    opt = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
  return learning_rate, opt
  
def get_train(split_train0_imgs, split_train0_labels, split_train0_ages, split_train1_imgs, split_train1_labels, split_train1_ages, num_gpus, weight_decay, opt):
  tower_grads = []
  total_loss_list_train = []
  cls_loss_list_train = []
  with tf.variable_scope(tf.get_variable_scope()):
    for i in xrange(num_gpus):
      with tf.device('/device:GPU:%d' % i):
        train_imgs = tf.concat([split_train0_imgs[i], split_train1_imgs[i]], axis=0)
        train_labels = tf.concat([split_train0_labels[i], split_train1_labels[i]], axis=0)
        train_ages = tf.concat([split_train0_ages[i], split_train1_ages[i]], axis=0)
        # load the model and get training losses
        logits, logits_ages, activations = ResNetGN().get_model(train_imgs, train_ages, training=True)        
        if 'sigmoid' not in exp_name:
          cls_loss = tf.losses.softmax_cross_entropy(tf.one_hot(train_labels,2), logits,label_smoothing=FLAGS.label_smoothing)
        elif 'sigmoid' in exp_name:
          cls_loss = tf.losses.sigmoid_cross_entropy(train_labels, logits,label_smoothing=FLAGS.label_smoothing)
        if weight_decay != None:
          if i == 0:
            rgl_loss = weight_decay * tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])
          total_loss = cls_loss + rgl_loss
        else: 
          total_loss = cls_loss

        # reuse variables for the next tower
        tf.get_variable_scope().reuse_variables()

        grads = opt.compute_gradients(total_loss, colocate_gradients_with_ops=True)
        tower_grads.append(grads)

        total_loss_list_train.append(total_loss)
        cls_loss_list_train.append(cls_loss)
  grads = average_gradients(tower_grads)
  total_loss_train = tf.math.reduce_mean(total_loss_list_train)
  cls_loss_train = tf.math.reduce_mean(cls_loss_list_train)
  
  age_loss_train = 0
  return grads, total_loss_train, cls_loss_train, age_loss_train, rgl_loss, train_imgs, train_labels, activations

def get_val(split_valall_imgs, split_valall_labels, split_valall_ages, num_gpus):
  loss_list_val = []
  labels_list_val = []
  softmax_list_val = []
  logits_list_val = []
  logits_ages_list_val = []

  # get the logits for validation set
  with tf.variable_scope(tf.get_variable_scope(), reuse=True):
    for i in xrange(num_gpus):
      with tf.device('/device:GPU:%d' % i):
        val_imgs = split_valall_imgs[i]
        val_labels = split_valall_labels[i]
        val_ages = split_valall_ages[i]
        logits_val, logits_ages_val, _ = ResNetGN().get_model(val_imgs, val_ages, training=False)
        
        if 'sigmoid' not in exp_name:
          cls_loss_val = tf.losses.softmax_cross_entropy(tf.one_hot(val_labels,2), logits_val)
          softmax_val = tf.nn.softmax(logits_val)[:,:,1]
        else:
          cls_loss_val = tf.losses.sigmoid_cross_entropy(val_labels, logits_val)
          softmax_val = tf.nn.sigmoid(logits_val)

        logits_ages_list_val.append(logits_ages_val)
        loss_list_val.append(cls_loss_val)
        labels_list_val.append(val_labels)
        softmax_list_val.append(softmax_val)
        logits_list_val.append(logits_val)
  labels_list_val = tf.reshape(tf.stack(labels_list_val),[-1])
  softmax_list_val = tf.reshape(tf.stack(softmax_list_val),[-1])
  val_loss = tf.math.reduce_mean(loss_list_val)
  val_age_loss = 0
  return softmax_list_val, labels_list_val, val_loss, val_age_loss, val_labels, logits_list_val


