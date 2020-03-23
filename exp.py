import sys
import os
import tensorflow as tf

model_folder = 'models_save/'
if 'lr_search' in sys.argv[2]:
  model_folder = model_folder + 'lr_grid_search/'

exp_name = sys.argv[1]

if 'train' in sys.argv[0]:
  if_train = True
  if 'continue_latest' in sys.argv[2]:
    if_continuetraining = True
    if_continuelatest = True
    decay_epoch = int(sys.argv[2].split('_')[-2]) 
    num_epoch = int(sys.argv[2].split('_')[-1])
  elif sys.argv[2] == 'cyclicallr':
    if_continuetraining = False
    if_continuelatest = False
    decay_epoch = 300
    num_epoch = 300
  elif sys.argv[2] == 'lr_search':
    if_continuetraining = False
    if_continuelatest = False
    decay_epoch = 50
    num_epoch = 50
elif 'eval' in sys.argv[0]:
  if_train = False 
  best_iter = int(sys.argv[2].split('_')[-1])     
  
### resolution
if '512' in exp_name:
  tf.app.flags.DEFINE_string(
  'dataset_dir', 'data_preprocessed_shards/', 'The directory where the dataset files are stored.')
  tf.app.flags.DEFINE_integer(
  'img_height', 632, 'The image height.')
  tf.app.flags.DEFINE_integer(
  'img_width', 512, 'The image width.')
  tf.app.flags.DEFINE_integer(
  'batch_size', 32, 'The number of samples in each batch.')
  tf.app.flags.DEFINE_integer(
  'num_gpus', 4, 'The number of gpus to be used.')

### optimizer
if if_train:
  if 'cyclicallr' in exp_name:
    tf.app.flags.DEFINE_string(
    'optimizer', 'cyclicallr', 'The name of the optimizer, one of "sgd" or "adam"')
    tf.app.flags.DEFINE_integer(
    'decay_epoch', None, 'Define when to delay learning rate.')
    tf.app.flags.DEFINE_float(
    'learning_rate_scratch', None, 'Learning rate for scratch model.')  
  elif 'lr0' in exp_name:
    tf.app.flags.DEFINE_string(
    'optimizer', 'sgd', 'The name of the optimizer, one of "sgd" or "adam"')
    tf.app.flags.DEFINE_integer(
    'decay_epoch', decay_epoch, 'Define when to delay learning rate.')
    tf.app.flags.DEFINE_float(
    'learning_rate_scratch', 0.00001, 'Learning rate for scratch model.')
  elif 'lr1' in exp_name:
    tf.app.flags.DEFINE_string(
    'optimizer', 'sgd', 'The name of the optimizer, one of "sgd" or "adam"')  
    tf.app.flags.DEFINE_integer(
    'decay_epoch', decay_epoch, 'Define when to delay learning rate.')
    tf.app.flags.DEFINE_float(
    'learning_rate_scratch', 0.0001, 'Learning rate for scratch model.')
  elif 'lr2' in exp_name:
    tf.app.flags.DEFINE_string(
    'optimizer', 'sgd', 'The name of the optimizer, one of "sgd" or "adam"')
    tf.app.flags.DEFINE_integer(
    'decay_epoch', decay_epoch, 'Define when to delay learning rate.')
    tf.app.flags.DEFINE_float(
    'learning_rate_scratch', 0.001, 'Learning rate for scratch model.') 
  elif 'lr3' in exp_name:
    tf.app.flags.DEFINE_string(
    'optimizer', 'sgd', 'The name of the optimizer, one of "sgd" or "adam"')
    tf.app.flags.DEFINE_integer(
    'decay_epoch', decay_epoch, 'Define when to delay learning rate.')
    tf.app.flags.DEFINE_float(
    'learning_rate_scratch', 0.01, 'Learning rate for scratch model.')
  elif 'lr4' in exp_name:
    tf.app.flags.DEFINE_string(
    'optimizer', 'sgd', 'The name of the optimizer, one of "sgd" or "adam"')
    tf.app.flags.DEFINE_integer(
    'decay_epoch', decay_epoch, 'Define when to delay learning rate.')
    tf.app.flags.DEFINE_float(
    'learning_rate_scratch', 0.1, 'Learning rate for scratch model.')
  elif 'lr5' in exp_name:
    tf.app.flags.DEFINE_string(
    'optimizer', 'sgd', 'The name of the optimizer, one of "sgd" or "adam"')
    tf.app.flags.DEFINE_integer(
    'decay_epoch', decay_epoch, 'Define when to delay learning rate.')
    tf.app.flags.DEFINE_float(
    'learning_rate_scratch', 5e-5, 'Learning rate for scratch model.')
  elif 'adam' in exp_name:
    tf.app.flags.DEFINE_string(
    'optimizer', 'adam', 'The name of the optimizer, one of "sgd" or "adam"')
    tf.app.flags.DEFINE_integer(
    'decay_epoch', 10, 'Define when to delay learning rate.')
    tf.app.flags.DEFINE_float(
    'learning_rate_scratch', 0.0003, 'Learning rate for scratch model.')

### checkpoint 
if if_train:
  if not if_continuetraining:
    if 'scratch' in exp_name:
      tf.app.flags.DEFINE_string(
      'checkpoint_path', None, 'The path to a checkpoint from which to fine-tune.')
    elif 'imagenet' in exp_name:
      tf.app.flags.DEFINE_string(
      'checkpoint_path', 'resnet50.ckpt', 'The path to a checkpoint from which to fine-tune.')
  elif if_continuetraining:
    if if_continuelatest:
      latest_path = tf.train.latest_checkpoint(os.path.join(model_folder, exp_name))
      if latest_path == None:
        checkpoint_path = 'resnet50.ckpt'
      else: 
        checkpoint_path = latest_path
      tf.app.flags.DEFINE_string(
      'checkpoint_path', checkpoint_path, 'The path to a checkpoint from which to fine-tune.')
elif not if_train:
  tf.app.flags.DEFINE_string(
  'checkpoint_path', os.path.join(model_folder, exp_name, 'my-model-'+str(best_iter)), 'The path to a checkpoint from which to fine-tune.')  

### dataset
if 'network3' in exp_name:
  tf.app.flags.DEFINE_string(
  'trainpos_dir1', 'network1_trainpos_shards', 'The directory where trainpos is stored.')
  tf.app.flags.DEFINE_string(
  'trainpos_dir2', 'network2_trainpos_shards', 'The directory where trainpos is stored.')
  tf.app.flags.DEFINE_string(
  'trainneg_dir', 'network_trainneg_shards', 'The directory where trainneg is stored.')
  tf.app.flags.DEFINE_string(
  'valall_dir', 'network3_valall_shards', 'The directory where valpos is stored.')
  tf.app.flags.DEFINE_integer(
  'train_samples', 9941, 'The min number of training samples per class.')
  tf.app.flags.DEFINE_integer(
  'valall_samples', 3008, 'The total number of validation samples per class.')
elif 'network1' in exp_name:
  tf.app.flags.DEFINE_string(
  'trainpos_dir', 'network1_trainpos_shards', 'The directory where trainpos is stored.')
  tf.app.flags.DEFINE_string(
  'trainneg_dir', 'network_trainneg_shards', 'The directory where trainneg is stored.')
  tf.app.flags.DEFINE_string(
  'valall_dir', 'network1_valall_shards', 'The directory where valpos is stored.')
  tf.app.flags.DEFINE_integer(
  'train_samples', 3079, 'The min number of training samples per class.')
  tf.app.flags.DEFINE_integer(
  'valall_samples', 1624, 'The total number of validation samples per class.')
elif 'network2' in exp_name:
  tf.app.flags.DEFINE_string(
  'trainpos_dir', 'network2_trainpos_shards', 'The directory where trainpos is stored.')
  tf.app.flags.DEFINE_string(
  'trainneg_dir', 'network_trainneg_shards', 'The directory where trainneg is stored.')
  tf.app.flags.DEFINE_string(
  'valall_dir', 'network2_valall_shards', 'The directory where valpos is stored.')
  tf.app.flags.DEFINE_integer(
  'train_samples', 6862, 'The min number of training samples per class.')
  tf.app.flags.DEFINE_integer(
  'valall_samples', 2748, 'The total number of validation samples per class.')

if 'network' in exp_name:
  tf.app.flags.DEFINE_string(
  'test_dir', 'network_testall_shards', 'The directory where test is stored.')
  tf.app.flags.DEFINE_integer(
  'test_samples', 6436, 'The number of test samples in total.')

### epoch number
if if_train:
  if '4epochs' in exp_name:
    tf.app.flags.DEFINE_integer(  
    'num_epochs', 4, 'The number of training epochs.')
  elif '3epochs' in exp_name:
    tf.app.flags.DEFINE_integer(
    'num_epochs', 3, 'The number of training epochs.')
  elif '2epochs' in exp_name:
    tf.app.flags.DEFINE_integer(
    'num_epochs', 2, 'The number of training epochs.')
  else:
     tf.app.flags.DEFINE_integer(
    'num_epochs', num_epoch, 'The number of training epochs.')

### random seed
if 'seed1' in exp_name:
  tf.app.flags.DEFINE_integer(
  'random_seed', 6111, 'The number of samples in each batch.') 
elif 'seed2' in exp_name:
  tf.app.flags.DEFINE_integer(
  'random_seed', 6112, 'The number of samples in each batch.')
elif 'seed3' in exp_name:
  tf.app.flags.DEFINE_integer(
  'random_seed', 6113, 'The number of samples in each batch.')
elif 'seed4' in exp_name:
  tf.app.flags.DEFINE_integer(
  'random_seed', 6114, 'The number of samples in each batch.')
elif 'seed5' in exp_name:
  tf.app.flags.DEFINE_integer(
  'random_seed', 6115, 'The number of samples in each batch.')

### others
tf.app.flags.DEFINE_string(
  'model_dir', os.path.join(model_folder, exp_name),
  'Directory where checkpoints and event logs are written to.')

### weight decay
if '_wd1' in exp_name:
  tf.app.flags.DEFINE_float(
  'weight_decay', 1e-8, 'The weight decay on the model weights.')
elif '_wd2' in exp_name:
  tf.app.flags.DEFINE_float(
  'weight_decay', 1e-7, 'The weight decay on the model weights.')
elif '_wd3' in exp_name:
  tf.app.flags.DEFINE_float(
  'weight_decay', 1e-6, 'The weight decay on the model weights.')
elif '_wd4' in exp_name:
  tf.app.flags.DEFINE_float(
  'weight_decay', 1e-5, 'The weight decay on the model weights.')
elif '_wd5' in exp_name:
  tf.app.flags.DEFINE_float(
  'weight_decay', 1e-4, 'The weight decay on the model weights.')
elif '_wd6' in exp_name:
  tf.app.flags.DEFINE_float(
  'weight_decay', 1e-3, 'The weight decay on the model weights.')
elif '_wd7' in exp_name:
  tf.app.flags.DEFINE_float(
  'weight_decay', 1e-2, 'The weight decay on the model weights.')
else:
  tf.app.flags.DEFINE_float(
  'weight_decay', 0, 'The weight decay on the model weights.')

if 'labelsmoothing0' in exp_name:
  tf.app.flags.DEFINE_float(
  'label_smoothing', 0.01, 'The weight decay on the model weights.')
elif 'labelsmoothing1' in exp_name:
  tf.app.flags.DEFINE_float(
  'label_smoothing', 0.005, 'The weight decay on the model weights.')
elif 'labelsmoothing2' in exp_name:
  tf.app.flags.DEFINE_float(
  'label_smoothing', 0.001, 'The weight decay on the model weights.')
elif 'labelsmoothing3' in exp_name:
  tf.app.flags.DEFINE_float(
  'label_smoothing', 0.0001, 'The weight decay on the model weights.')
else:
  tf.app.flags.DEFINE_float(
  'label_smoothing', 0, 'The weight decay on the model weights.')

FLAGS = tf.app.flags.FLAGS
print(tf.app.flags.FLAGS.flag_values_dict())
