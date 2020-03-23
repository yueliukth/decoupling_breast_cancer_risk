import tensorflow as tf
import pandas as pd
import numpy as np
import os
import glob
import sys
import threading
from datetime import datetime

csv_file = 'data_csv.csv'
full_path = 'data_preprocessed/'
full_path_out = 'data_preprocessed_shards/'
data_split_list = ['network1_trainpos', 'network_trainneg', 'network1_valall', 'network_testall', 'network2_trainpos', 'network2_valall', 'network3_valall']

tf.app.flags.DEFINE_string('input_directory', full_path,
  'input png images directory')

tf.app.flags.DEFINE_string('output_directory', full_path_out,
  'output TFRecord shards directory')

tf.app.flags.DEFINE_string('csv_file', csv_file,
  'path to csv file containing the medical records')

tf.app.flags.DEFINE_integer('num_shards', 240,
   'Number of shards')

tf.app.flags.DEFINE_integer('num_threads', 12,
   'Number of threads to preprocess the images.')

tf.app.flags.DEFINE_integer('nan_value', -9999,
   'value to be used instead of nan')

FLAGS = tf.app.flags.FLAGS

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
  """Wrapper for inserting float features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))
 
def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""
  def __init__(self):
    # create a single session to run all image coding calls
    self._sess = tf.Session()
    # initialize function that decodes RGB PNG data
    self._decode_png_data = tf.placeholder(dtype=tf.string)
    self._decode_png = tf.image.decode_png(self._decode_png_data, channels=3)
  def decode_png(self, image_data):
    image = self._sess.run(self._decode_png, feed_dict={self._decode_png_data:image_data})
    assert len(image.shape) == 3
    return image

def _convert_to_example(filename, image_buffer, label, height, width, medical_dict):
  """Build an Example proto for an example.

  Args:
  filename: string, path to an image file, e.g., '/path/to/example.JPG'
  image_buffer: string, PNG encoding of RGB image
  label: integer, identifier for the ground truth for the network
  height: integer, image height in pixels
  width: integer, image width in pixels
  medical_dict: a dictionary/series with all the patient records
  Returns:
  Example proto
  """
  colorspace = 'RGB'
  channels = 3
  image_format = 'PNG'
  example = tf.train.Example(features=tf.train.Features(feature={
    'image/height': _int64_feature(height),
    'image/width': _int64_feature(width),
    'image/colorspace': _bytes_feature(tf.compat.as_bytes(colorspace)),
    'image/channels': _int64_feature(channels),
    'image/format': _bytes_feature(tf.compat.as_bytes(image_format)),
    'image/class/label': _int64_feature(label),
    'image/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
    'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer, encoding='utf-16')),
    'medical/basename': _bytes_feature(tf.compat.as_bytes(medical_dict['basename'])),
    'medical/age': _float_feature(float(medical_dict['x_age_mammo'])),
    'medical/thickness': _float_feature(float(medical_dict['dicom_bodypartthickness'])),
    'medical/compression': _float_feature(float(medical_dict['dicom_compressionforce'])),
    'medical/exposure': _float_feature(float(medical_dict['dicom_exposuretime'])),
    'medical/current': _float_feature(float(medical_dict['dicom_xraytubecurrent'])), 
}))
  return example

def _process_image(filename, coder):
  """Process a single image file.

  Args:
  filename: string, path to an image file e.g., '/path/to/example.JPG'.
  coder: instance of ImageCoder to provide TensorFlow image coding utils. 
  Returns:
  image_buffer: string, JPEG encoding of RGB image.
  height: integer, image height in pixels.
  width: integer, image width in pixels.
  """
  # read the image file
  with tf.gfile.FastGFile(filename,'r') as f:
    image_data = f.read()
  # decode the RGB png
  image = coder.decode_png(image_data)

  # convert grayscale to RGB
  if len(image.shape) == 2:
    image = np.tile(np.reshape(image, np.concatenate((image.shape, [1]))), [1,1,3])
  
  # check that image converted to RGB
  assert len(image.shape) == 3
  height = image.shape[0]
  width = image.shape[1]
  assert image.shape[2] == 3
  return image_data, height, width

def  _process_image_files_batch(coder, thread_index, ranges, name, filenames, labels, medical_dicts, num_shards, folder_path):
  """Processes and saves list of images as TFRecord in 1 thread.

  Args:
  coder: instance of ImageCoder to provide TensorFlow image coding utils.
  thread_index: integer, unique batch to run index is within [0, len(ranges)).
  ranges: list of pairs of integers specifying ranges of each batches to analyze in parallel.
  name: string, unique identifier specifying the data set
  filenames: list of strings; each string is a path to an image file
  labels: list of integer; each integer identifies the ground truth
  num_shards: integer number of shards for this data set.
  """
  # each thread produces N shards where N = int(num_shards / num_threads)
  # for instance, if num_shards = 128, and the num_threads = 2, then the first thread would produce shards [0, 64)

  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1], num_shards_per_batch + 1).astype(int)
  num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  counter = 0
  for s in range(num_shards_per_batch):
    # generate a sharded version of the file name, eg, 'trainpos-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
    output_file = os.path.join(folder_path, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)

    for i in files_in_shard:
      filename = filenames[i]
      label = labels[i]
      medical_dict = medical_dicts.iloc[i, :]

      image_buffer, height, width = _process_image(filename, coder)
      example = _convert_to_example(filename, image_buffer, label, height, width, medical_dict)
      writer.write(example.SerializeToString())
      shard_counter += 1
      counter += 1

      if not counter % 1000:
        print('%s [thread %d]: Processed %d of %d images in thread batch.' % (datetime.now(), thread_index, counter, num_files_in_thread))
        sys.stdout.flush()

    writer.close()
    print('%s [thread %d]: Wrote %d images to %s' % (datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()

    shard_counter = 0
  print('%s [thread %d]: Wrote %d images to %d shards.' % (datetime.now(), thread_index, counter, num_files_in_thread))
  sys.stdout.flush()

def _process_image_files(name, filenames, labels, medical_dicts, num_shards, folder_path):
  """Process and save list of images as TFRecord of Example protos.

  Args:
  name: string, unique identifier specifying the data set
  filenames: list of strings; each string is a path to an image file
  labels: list of integer; each integer identifies the ground truth
  num_shards: integer number of shards for this data set.
  """

  assert len(filenames) == len(labels)
  assert len(filenames) == medical_dicts.shape[0]

  # break all images into batches
  spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
  ranges = []
  for i in range(len(spacing)-1):
    ranges.append([spacing[i], spacing[i+1]])

  # lauch a thread for each batch
  print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
  sys.stdout.flush()

  # create a mechnism for monitoring when all threads are finished
  coord = tf.train.Coordinator()

  # create a generic Tensorflow-based utility for converting all image codings
  coder = ImageCoder()

  threads = []
  for thread_index in range(len(ranges)):
    args = (coder, thread_index, ranges, name, filenames, labels, medical_dicts, num_shards, folder_path)
    t = threading.Thread(target=_process_image_files_batch, args=args)
    t.start()
    threads.append(t)

  # wait for all the threads to terminate
  coord.join(threads)
  print('%s: Finished writing all %d images in data set.' % (datetime.now(), len(filenames)))
  sys.stdout.flush()

def _process_dataset(name, csv_file, num_shards, folder_path, print_box):
  """Process a complete data set and save it as a TFRecord.

  Args:
  name: string, unique identifier specifying the data set.
  directory: string, root path to the data set.
  num_shards: integer number of shards for this data set.
  """
  df = pd.read_csv(csv_file, delimiter=';', dtype={'sourcefile':str})
  count = 0 
  filenames = [] 
  inds = []
  labels = []
  for i in range(df.shape[0]):
    row = df.loc[i, :]
    basename = row['basename'][:-3]+'png'
    img_path = os.path.join(full_path, basename)
   if row[name] == 1 and os.path.exists(img_path):
      count += 1 
      filenames.append(img_path)
      inds.append(i)
      labels.append(row[name+'_label'])
     
  print_box.append([str(sum(labels)), str(count - sum(labels)), '/'])        
  print(str(count), 'images are being processed.')
  print('number of pos: ', str(sum(labels)))
  print('number of neg: ', str(count - sum(labels)))
  
  medical_dicts = df.iloc[inds,:]
  medical_dicts.index = range(medical_dicts.shape[0])
  medical_dicts = medical_dicts.fillna(FLAGS.nan_value)

  _process_image_files(name, filenames, labels, medical_dicts, num_shards, folder_path)
  return print_box

def main(unused_argv):
  assert not FLAGS.num_shards % FLAGS.num_threads, (
    'Please make the FLAGS.num_threads commensurate with FLAGS.num_shards')
  print_box = []

  for split in data_split_list: 
    folder_path = os.path.join(FLAGS.output_directory, split) + '_shards'
    print('Saving results to %s' % folder_path)
    if not os.path.exists(folder_path):
      os.makedirs(folder_path)
    print_box = _process_dataset(split, FLAGS.csv_file, FLAGS.num_shards, folder_path, print_box)
    print(data_split_list)
    print(print_box)

if __name__ == "__main__":
  tf.app.run()

