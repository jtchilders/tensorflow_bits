import glob
import tensorflow as tf
import numpy as np


# get files
filelist = glob.glob('/Users/jchilders/git/atlas-yolo/data/*.npz')

# a generator function to serve data from one file
def get_data(filename):
   print('filename = %s' % filename)
   npdata = np.load(filename)
   features = npdata['raw']
   labels = npdata['truth']
   assert features.shape[0] == labels.shape[0]
   for i in range(features.shape[0]):
      yield features[i],labels[i]


# create dataset of filenames
ds = tf.data.Dataset.from_tensor_slices(filelist)
# use interleave to create multiple threads that feed files
# problem is these threads read the same number of events from
# each file in round robin style so each are exhausted at the same
# time and therefore the file reading all happens in sync, kinda
# negates the idea...
ds = ds.interleave(lambda filename: tf.data.Dataset.from_generator(get_data,(tf.float64,tf.int64),(tf.TensorShape([ 16, 256, 9600]), tf.TensorShape([1,10])),args=([filename])),
      6,1,3)
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
