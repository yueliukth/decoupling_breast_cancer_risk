import os
import time
import math
import numpy as np
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import random
from exp import *

def color(x):
#  x = tf.image.random_hue(x, 0.08)
#  x = tf.image.random_saturation(x, 0.6, 1.6)
  x = tf.image.random_brightness(x, 0.05)
  x = tf.image.random_contrast(x, 0.7, 1.3)
  return x

def img_train(image):  
  img_height = FLAGS.img_height
  img_width = FLAGS.img_width
  image = tf.reshape(image, [img_height, img_width, 1])
  if image.dtype != tf.float32:
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

  # random rotation
  image = tf.contrib.image.rotate(image, tf.random_uniform([1], minval=-10, maxval=10, seed=FLAGS.random_seed) * math.pi / 180)
  # random crop 
  image = tf.image.resize_image_with_crop_or_pad(image, int(img_height*1.125), int(img_width*1.125))
  image = tf.random_crop(image, [img_height, img_width, 1], seed=FLAGS.random_seed)

  # random flipping 
  if 'randomflip' in exp_name:
    image = tf.image.random_flip_left_right(image,seed=FLAGS.random_seed)
  if 'randomcolor' in exp_name:
    image = color(image)  
  if 'randomcontrast' in exp_name:
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5, seed=FLAGS.random_seed)

  if 'standardization' in exp_name:
    image = tf.image.per_image_standardization(image)
  
  image = tf.tile(image, [1,1,3])

  return image

def img_val(image): 
  img_height = FLAGS.img_height
  img_width = FLAGS.img_width
  image = tf.reshape(image, [img_height, img_width, 1])
  if image.dtype != tf.float32:
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

  # centeral crop
  image = tf.image.resize_image_with_crop_or_pad(image, img_height, img_width)
  image = tf.image.per_image_standardization(image)
  image = tf.tile(image, [1,1,3])
  return image

def _parser(record):
  keys_to_features = {
    'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
    'image/class/label': tf.FixedLenFeature([1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
    'medical/age': tf.FixedLenFeature([1], tf.float32, default_value=tf.zeros([1], dtype=tf.float32))
    }
  parsed = tf.parse_single_example(record, keys_to_features)
  image = tf.image.decode_image(parsed['image/encoded'],dtype=tf.uint16)
  label = tf.cast(parsed['image/class/label'],tf.int64)
  age = tf.cast(parsed['medical/age'],tf.float32)
  return image, label, age

def combine_fn(record, if_training):
  image, label, age = _parser(record)
  if if_training:
    image = img_train(image)
  else:
    image = img_val(image)
  return image, label, age

def _get_data(filenames, batch_perclass, if_training):
  dataset = tf.data.TFRecordDataset(filenames)
  if if_training:
    dataset = dataset.shuffle(int(FLAGS.batch_size/2) * 20, seed=FLAGS.random_seed)
  dataset = dataset.repeat()
  if if_training:
    dataset = dataset.map(lambda record: combine_fn(record, if_training=True))
  else:
    dataset = dataset.map(lambda record: combine_fn(record, if_training=False))
  dataset = dataset.batch(batch_perclass)
  dataset = dataset.prefetch(1)
  iterator = dataset.make_initializable_iterator()
  features, labels, ages = iterator.get_next()
  return features, labels, ages, iterator

def data_prep_network3(dataset_dir, trainpos_dir1, trainpos_dir2, trainneg_dir, valall_dir, num_gpus):
  train0_filenames = [os.path.join(dataset_dir, trainneg_dir, name) for name in os.listdir(os.path.join(dataset_dir, trainneg_dir))]
  train1_filenames = [os.path.join(dataset_dir, trainpos_dir1, name) for name in os.listdir(os.path.join(dataset_dir, trainpos_dir1))]+[os.path.join(dataset_dir, trainpos_dir2, name) for name in os.listdir(os.path.join(dataset_dir, trainpos_dir2))]
  valall_filenames = [os.path.join(dataset_dir, valall_dir, name) for name in os.listdir(os.path.join(dataset_dir, valall_dir))]

  train0_imgs, train0_labels, train0_ages, train0_iterator = _get_data(train0_filenames, int(FLAGS.batch_size/2), if_training=True)
  train1_imgs, train1_labels, train1_ages, train1_iterator = _get_data(train1_filenames, int(FLAGS.batch_size/2),  if_training=True)
  valall_imgs, valall_labels, valall_ages, valall_iterator = _get_data(valall_filenames, int(FLAGS.batch_size), if_training=False)
  split_train0_imgs = tf.split(value=train0_imgs, num_or_size_splits=num_gpus, axis=0)
  split_train0_labels = tf.split(value=train0_labels, num_or_size_splits=num_gpus, axis=0)
  split_train0_ages = tf.split(value=train0_ages, num_or_size_splits=num_gpus, axis=0)
  split_train1_imgs = tf.split(value=train1_imgs, num_or_size_splits=num_gpus, axis=0)
  split_train1_labels = tf.split(value=train1_labels, num_or_size_splits=num_gpus, axis=0)
  split_train1_ages = tf.split(value=train1_ages, num_or_size_splits=num_gpus, axis=0)
  split_valall_imgs = tf.split(value=valall_imgs, num_or_size_splits=num_gpus, axis=0)
  split_valall_labels = tf.split(value=valall_labels, num_or_size_splits=num_gpus, axis=0)
  split_valall_ages = tf.split(value=valall_ages, num_or_size_splits=num_gpus, axis=0)

  return split_train0_imgs, split_train0_labels, split_train0_ages, train0_iterator, split_train1_imgs, split_train1_labels, split_train1_ages, train1_iterator, split_valall_imgs, split_valall_labels, split_valall_ages, valall_iterator

def data_prep_network(dataset_dir, trainpos_dir, trainneg_dir, valall_dir, num_gpus):
  train0_filenames = [os.path.join(dataset_dir, trainneg_dir, name) for name in os.listdir(os.path.join(dataset_dir, trainneg_dir))]
  train1_filenames = [os.path.join(dataset_dir, trainpos_dir, name) for name in os.listdir(os.path.join(dataset_dir, trainpos_dir))]
  valall_filenames = [os.path.join(dataset_dir, valall_dir, name) for name in os.listdir(os.path.join(dataset_dir, valall_dir))]

  train0_imgs, train0_labels, train0_ages, train0_iterator = _get_data(train0_filenames, int(FLAGS.batch_size/2), if_training=True)
  train1_imgs, train1_labels, train1_ages, train1_iterator = _get_data(train1_filenames, int(FLAGS.batch_size/2),  if_training=True)
  valall_imgs, valall_labels, valall_ages, valall_iterator = _get_data(valall_filenames, int(FLAGS.batch_size), if_training=False)

  split_train0_imgs = tf.split(value=train0_imgs, num_or_size_splits=num_gpus, axis=0)
  split_train0_labels = tf.split(value=train0_labels, num_or_size_splits=num_gpus, axis=0)
  split_train0_ages = tf.split(value=train0_ages, num_or_size_splits=num_gpus, axis=0)
  split_train1_imgs = tf.split(value=train1_imgs, num_or_size_splits=num_gpus, axis=0)
  split_train1_labels = tf.split(value=train1_labels, num_or_size_splits=num_gpus, axis=0)
  split_train1_ages = tf.split(value=train1_ages, num_or_size_splits=num_gpus, axis=0)
  split_valall_imgs = tf.split(value=valall_imgs, num_or_size_splits=num_gpus, axis=0)
  split_valall_labels = tf.split(value=valall_labels, num_or_size_splits=num_gpus, axis=0)
  split_valall_ages = tf.split(value=valall_ages, num_or_size_splits=num_gpus, axis=0)
  return split_train0_imgs, split_train0_labels, split_train0_ages, train0_iterator, split_train1_imgs, split_train1_labels, split_train1_ages, train1_iterator, split_valall_imgs, split_valall_labels, split_valall_ages, valall_iterator

