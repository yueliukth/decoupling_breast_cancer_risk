import os
import time
import math
import numpy as np
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import random
from exp import *
from tensorflow.python.ops import variables
from model_functions import *

def _parser_val(record):
  keys_to_features = {
    'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
    'image/class/label': tf.FixedLenFeature([1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
    'medical/basename': tf.FixedLenFeature((), tf.string, default_value='') 
    }
  parsed = tf.parse_single_example(record, keys_to_features)
  image = tf.image.decode_image(parsed['image/encoded'])
  img_height = FLAGS.img_height
  img_width = FLAGS.img_width
  image = tf.reshape(image, [img_height, img_width, 1])

  if image.dtype != tf.float32:
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

  # centeral crop
  image = tf.image.resize_image_with_crop_or_pad(image, img_height, img_width)

  image = tf.image.per_image_standardization(image)
  image = tf.tile(image, [1,1,3])
  label = tf.cast(parsed['image/class/label'],tf.int64)
  basename = parsed['medical/basename']

  return image, label, basename

def _get_data(filenames):
  dataset = tf.data.TFRecordDataset(filenames)
  dataset = dataset.cache()    
  dataset = dataset.repeat(2) 
  dataset = dataset.map(_parser_val, num_parallel_calls=30)
  dataset = dataset.batch(FLAGS.batch_size)
  dataset = dataset.prefetch(10)
  iterator = dataset.make_initializable_iterator()
  features, labels, basenames = iterator.get_next()
  return features, labels, basenames, iterator

def main(_):
  start = time.time()
  tf.logging.set_verbosity(tf.logging.INFO)
  
  with tf.Graph().as_default(), tf.device('/device:CPU:0'):  
    tf.set_random_seed(1234)
    
    test_filenames = [os.path.join(FLAGS.dataset_dir, FLAGS.test_dir, name) for name in os.listdir(os.path.join(FLAGS.dataset_dir, FLAGS.test_dir))]

    test_imgs, test_labels, test_basenames, test_iterator = _get_data(test_filenames)
    split_test_imgs = tf.split(value=test_imgs, num_or_size_splits=FLAGS.num_gpus, axis=0)
    split_test_labels = tf.split(value=test_labels, num_or_size_splits=FLAGS.num_gpus, axis=0)
    split_test_basenames = tf.split(value=test_basenames, num_or_size_splits=FLAGS.num_gpus, axis=0)
    
    loss_list_test = []
    labels_list_test = []
    softmax_list_test = []
    basename_list_test = []
 
    # get the logits for validation set
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      for i in xrange(FLAGS.num_gpus):
        with tf.device('/device:GPU:%d' % i): 
          logits_test, ages_test, _ = ResNetGN().get_model(split_test_imgs[i], split_test_labels[i], training=False) 
          if 'sigmoid' not in exp_name:
            softmax_test = tf.nn.softmax(logits_test)[:,:,1]
            cls_loss_test = tf.losses.softmax_cross_entropy(tf.one_hot(split_test_labels[i],2), logits_test)
          else:
            softmax_test = tf.nn.sigmoid(logits_test)
            cls_loss_test = tf.losses.sigmoid_cross_entropy(split_test_labels[i], logits_test)

          loss_list_test.append(cls_loss_test)
          labels_list_test.append(split_test_labels[i])
          softmax_list_test.append(softmax_test)
          basename_list_test.append(split_test_basenames[i])

    labels_list_test = tf.reshape(tf.stack(labels_list_test),[-1])
    softmax_list_test = tf.reshape(tf.stack(softmax_list_test),[-1])  
    basename_list_test = tf.reshape(tf.stack(basename_list_test),[-1])   
        
    test_loss = tf.math.reduce_mean(loss_list_test)
         
    config = tf.ConfigProto(log_device_placement=False,  allow_soft_placement=True)

    with tf.Session(config=config) as sess: 
      sess.run(test_iterator.initializer)

      init = tf.global_variables_initializer()
      sess.run(init)
      saver = tf.train.Saver()
      saver.restore(sess, FLAGS.checkpoint_path)  
      var_list = variables.global_variables()
      num_imgs = FLAGS.test_samples
      
      num_batches_test = int(math.ceil(num_imgs/FLAGS.batch_size))+1
      print 'Number of batches of test: ' + str(num_batches_test)

      loss_list = []
      y_scores = []
      y_trues = []
      basenames = []      

      for eval_iter in range(num_batches_test):
        print eval_iter
        softmax, labels, loss, basename = sess.run([softmax_list_test, labels_list_test, test_loss, basename_list_test]) 
        loss_list.append(loss)
        y_scores.extend(softmax)
        y_trues.extend(labels) 
        basenames.extend(basename)
      
      output_file_path = os.path.join(os.path.dirname(FLAGS.checkpoint_path), 'preds_test_' + str(best_iter) + '.txt')
     
      of = open(output_file_path, 'w')
      for i in range(len(basenames)):
        of.write('%s %.12f %d\n' % (basenames[i], y_scores[i], y_trues[i]))      

      avg_loss = sum(loss_list)/len(loss_list)
     
if __name__ == '__main__':
  tf.app.run()
