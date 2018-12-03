import glob
import tensorflow as tf
import numpy as np


# get files
filelist = glob.glob('/Users/jchilders/git/atlas-yolo/data/*.npz')


# take a string filename and open the compressed numpy files
# return features and labels
def get_data_from_filename(filename):
   print('filename = %s',filename)
   npdata = np.load(filename)
   return npdata['raw'],npdata['truth']

# wrapper function to make this a representative function that Tensorflow
# can add to a graph
def get_data_wrapper(filename):
   features, labels = tf.py_func(get_data_from_filename, [filename], (tf.float64, tf.int64))  
   return tf.data.Dataset.from_tensor_slices((features, labels))


# create dataset of filenames
ds = tf.data.Dataset.from_tensor_slices(filelist)
# map those filenames into feature+label elements, flatten between files to get one long list
ds = ds.flat_map(get_data_wrapper)
# set batch size
ds = ds.batch(2)

# create an interator
iter = ds.make_one_shot_iterator()
# get the iterator function
get_batch = iter.get_next()

# iterate over the files
with tf.Session() as sess:
   while True:
      x,y = sess.run(get_batch)

      print(x.shape)
      print(y.shape)

