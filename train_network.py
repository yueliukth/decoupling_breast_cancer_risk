import sys
import os
import time
import math
import numpy as np
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import random
from exp import *
from train_functions import *
from dataset_functions import *
from model_functions import *
from tensorflow.contrib.memory_stats import MaxBytesInUse

# each new session restarts the random state
os.environ['PYTHONHASHSEED']=str(FLAGS.random_seed)
os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
np.random.seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)
     
def main(_): 
  start = time.time()
  tf.logging.set_verbosity(tf.logging.INFO)
  
  # get number of iterations for each epoch
  steps_per_epoch = int(FLAGS.train_samples*2/FLAGS.batch_size)
  print 'steps_per_epoch is ' + str(steps_per_epoch)

  if not os.path.exists(FLAGS.model_dir):
    os.makedirs(FLAGS.model_dir)

  tf.reset_default_graph()

  with tf.Graph().as_default(), tf.device('/device:CPU:0'):  
    tf.set_random_seed(FLAGS.random_seed)

    global_step=tf.train.create_global_step()

    # define learning rate and optimizer
    learning_rate, opt = define_lr_opt(global_step, steps_per_epoch, FLAGS.optimizer, FLAGS.learning_rate_scratch, FLAGS.decay_epoch, FLAGS.num_epochs)

    # data preparation  
    if 'network3' in exp_name:
      split_train0_imgs, split_train0_labels, split_train0_ages, train0_iterator, split_train1_imgs, split_train1_labels, split_train1_ages, train1_iterator, split_valall_imgs, split_valall_labels, split_valall_ages, valall_iterator = data_prep_network3(FLAGS.dataset_dir, FLAGS.trainpos_dir1, FLAGS.trainpos_dir2, FLAGS.trainneg_dir, FLAGS.valall_dir, FLAGS.num_gpus)
    else:
      split_train0_imgs, split_train0_labels, split_train0_ages, train0_iterator, split_train1_imgs, split_train1_labels, split_train1_ages, train1_iterator, split_valall_imgs, split_valall_labels, split_valall_ages, valall_iterator = data_prep_network(FLAGS.dataset_dir, FLAGS.trainpos_dir, FLAGS.trainneg_dir, FLAGS.valall_dir, FLAGS.num_gpus)

    # define train process   
    grads, total_loss_train, cls_loss_train, age_loss_train, rgl_loss, train_imgs, train_labels, activations = get_train(split_train0_imgs, split_train0_labels, split_train0_ages, split_train1_imgs, split_train1_labels, split_train1_ages, FLAGS.num_gpus, FLAGS.age_option, FLAGS.weight_decay, opt)
    
    optimizer = opt.apply_gradients(grads, global_step=global_step)

    # get train summaries
    if FLAGS.weight_decay is not None:
      tf.summary.scalar('weight_decay', FLAGS.weight_decay)
      tf.summary.scalar('rgl_batch_loss', rgl_loss)
    tf.summary.scalar('total_batch_loss', total_loss_train)
    tf.summary.scalar('cls_batch_loss', cls_loss_train)
    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.image('train_cropped_image', train_imgs, 1)

    # define val process
    softmax_list_val, labels_list_val, val_loss, val_age_loss, val_labels, logits_val = get_val(split_valall_imgs, split_valall_labels, split_valall_ages, FLAGS.num_gpus, FLAGS.age_option) 

    # model savers
    model_vars = tf.trainable_variables()
  
    # create a saver
    if 'asgd' in exp_name:
      saver = opt.swapping_saver(max_to_keep=200)
    else:
      saver = tf.train.Saver(max_to_keep=200)  
   
    # merge all summaries
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    summary_op = tf.summary.merge(list(summaries), name='summary_op')
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=gpu_options)

    if FLAGS.checkpoint_path is not None:
      restored_vars  = get_tensors_in_checkpoint_file(file_name=FLAGS.checkpoint_path)
      tensors_to_load = build_tensors_in_checkpoint_file(restored_vars)
      saver1 = tf.train.Saver(tensors_to_load)
    
    # create the summary writer afrer graph definition and before running session
    val_writer = tf.summary.FileWriter(FLAGS.model_dir + '/val/')
    
    with tf.Session(config=config) as sess:
      train_writer = tf.summary.FileWriter(FLAGS.model_dir + '/train/', sess.graph)      
      # initialize the dataset and global variables
      sess.run(tf.global_variables_initializer())
      
      sess.run([train0_iterator.initializer, train1_iterator.initializer, valall_iterator.initializer])
      sess.run(tf.global_variables_initializer())

      if FLAGS.checkpoint_path is not None:
        saver1.restore(sess, FLAGS.checkpoint_path)
       
      print 'Starting training.' 
      sess.graph.finalize()
      tf.get_default_graph().finalize()
      
      init_global_step = sess.run(tf.train.get_global_step())
      for i in range(steps_per_epoch * FLAGS.num_epochs-init_global_step):
       
        global_step = sess.run(tf.train.get_global_step())
 
        if 'cyclicallr' in exp_name:
          train_summary, train_batch_loss = sess.run([summary_op, total_loss_train])
          train_writer.add_summary(train_summary, global_step)
          msg = "Optimization Iteration: {0:}, Training Batch Loss: {1:}"
          print msg.format(global_step, train_batch_loss)

        if i == 0 or global_step % steps_per_epoch == 0: 
          train_summary, train_batch_loss = sess.run([summary_op, total_loss_train])
          train_writer.add_summary(train_summary, global_step)
         
          msg = "Optimization Iteration: {0:}, Training Batch Loss: {1:}"
          print msg.format(global_step, train_batch_loss)
          print 'Time for each epoch:'
          print time.time() - start
          start = time.time()
          saver.save(sess, FLAGS.model_dir+'/my-model', global_step=global_step)
          print 'New model is saved for ' + str(global_step/steps_per_epoch) + ' epoch.'
          print 'Starting validation of ' + str(global_step/steps_per_epoch) + ' epoch.' 
          
          num_batches_val = int(math.ceil(FLAGS.valall_samples/FLAGS.batch_size))+1
          print 'Number of batches in val: ' + str(num_batches_val)

          loss_list = []
          y_scores = []
          y_trues = []
        
          for eval_iter in range(num_batches_val):
            softmax, labels, loss, logits = sess.run([softmax_list_val, labels_list_val, val_loss, logits_val]) 
            loss_list.append(loss)
            y_scores.extend(softmax)
            y_trues.extend(labels) 

          auc = roc_auc_score(y_trues, y_scores)          
          avg_loss = sum(loss_list)/len(loss_list)
          
          val_summary = tf.Summary()
          val_summary.value.add(tag="validation_auc", simple_value=auc)
          val_summary.value.add(tag="validation_average_cls_oss", simple_value=avg_loss)
          val_writer.add_summary(val_summary, global_step)
           
          msg_val = "Validation Iteration: {0:}, Validation Loss: {1:}, Validation AUC: {2:}"
          print msg_val.format(global_step, avg_loss, auc)
         
        _ = sess.run(optimizer)
      train_writer.close()      
      val_writer.close()
      
      print time.time() - start

if __name__ == '__main__':
  tf.app.run()
