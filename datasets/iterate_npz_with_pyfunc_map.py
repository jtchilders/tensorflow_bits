import glob,time
import tensorflow as tf
import numpy as np


# get files
filelist = glob.glob('/Users/jchilders/git/atlas-yolo/data/*.npz')
print(' files in list: %s' % len(filelist))


# take a string filename and open the compressed numpy files
# return features and labels
def get_data_from_filename(filename):
   print('reading filename %s' % filename)
   start = time.time()
   npdata = np.load(filename)
   features = npdata['raw']
   labels = npdata['truth']
   print('read filename %s in %10.2f' % (filename,time.time() - start))
   return features,labels


# wrapper function to make this a representative function that Tensorflow
# can add to a graph
def get_data_wrapper(filename):
   features, labels = tf.py_func(get_data_from_filename, [filename], (tf.float64, tf.int64))
   # return tf.data.Dataset.from_tensor_slices((features, labels))
   return (features,labels)


# create dataset of filenames
ds = tf.data.Dataset.from_tensor_slices(filelist)
# map those filenames into feature+label elements, using map to get parallel behavior
ds = ds.map(get_data_wrapper,num_parallel_calls=2)
# now flatten dataset
ds = ds.flat_map(lambda *x: tf.data.Dataset.from_tensor_slices(x))
# set batch size
ds = ds.batch(2)

# create an interator
iter = ds.make_one_shot_iterator()
# get the iterator function
get_batch = iter.get_next()

# iterate over the files
with tf.Session() as sess:
   while True:
      print('starting')
      start = time.time()
      x,y = sess.run(get_batch)

      print(x.shape)
      print(y.shape)
      print('run time: %10.2f' % (time.time() - start))

