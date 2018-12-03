from __future__ import division
import glob,time,sys
import cPickle as pickle
import tensorflow as tf
import numpy as np
import multiprocessing as mp
if sys.version_info[0] < 3:
   import Queue as queue
else:
   import queue

# this solution works, but the 'Queue.get()' call takes a long time.
# I suspect this is because the pickling/un-pickling of the numpy arrays

mgr = mp.Manager()


class MyGenerator(object):

   def __init__(self,feature_shape,label_shape):
      self.feature_shape = feature_shape
      self.label_shape = label_shape

      self.data = mgr.dict()
      self.data_lock = mgr.Lock()
      self.data_ready = mgr.Event()

   def serve_file(self,filename):
      features,labels = self.get_data(filename)
      i = 0
      while i < features.shape[0]:
         try:
            start = time.time()
            gen_queue.put((features[i],labels[i]),block=True,timeout=60)
            print('queue put time: %10.5f' % (time.time() - start))
            i += 1
         except queue.Full:
            print('queue is full, trying again')

   print('done with filename = %s' % filename)
   return filename


   def get_data(self,filename):
      print('filename = %s' % filename)
      npdata = np.load(filename)
      start = time.time()
      features = npdata['raw']
      labels = npdata['truth']
      print('data decompress time: %10.5f' % (time.time() - start))
      assert features.shape[0] == labels.shape[0]
      return features,labels


   def generate(self):
      # get files
      filelist = glob.glob('/Users/jchilders/git/atlas-yolo/data/*.npz')

      p = mp.Pool(5)
      enum = p.imap_unordered(serve_file,filelist)
      print('enum = %s' % enum)

      print('enum = %s' % dir(enum))
      p.close()
      while True:

         start = time.time()
         feature,label = gen_queue.get()
         print('[%d] queue get time: %10f' % (gen_counter,time.time() - start))
         yield feature,label





# create dataset of filenames
ds = tf.data.Dataset.from_generator(parallel_gen,(tf.float64,tf.int64),(tf.TensorShape([ 16, 256, 9600]), tf.TensorShape([1,10])))
ds = ds.batch(2)

iter = ds.make_one_shot_iterator()
get_batch = iter.get_next()

with tf.Session() as sess:
   while True:
      x,y = sess.run(get_batch)
      print('x shape: %s; y shape: %s' % (x.shape,y.shape))
