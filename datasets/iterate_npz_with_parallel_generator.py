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

qmax = 40
gen_queue = mp.Queue(qmax)


def get_data(filename):
   print('filename = %s' % filename)
   npdata = np.load(filename)
   start = time.time()
   features = npdata['raw']
   labels = npdata['truth']
   print('data decompress time: %10.5f' % (time.time() - start))
   assert features.shape[0] == labels.shape[0]
   return features,labels


def serve_file(filename):
   features,labels = get_data(filename)
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

gen_counter = 0

def parallel_gen():
   global gen_counter
   gen_counter += 1
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



def simple_gen():
   global gen_counter
   gen_counter += 1
   # get files
   filelist = glob.glob('/Users/jchilders/git/atlas-yolo/data/*.npz')

   for filename in filelist:
      features,labels = get_data(filename)

      for i in range(features.shape[0]):
         print('yield from gen %d' % gen_counter)
         yield features[i],labels[i]




# create dataset of filenames
ds = tf.data.Dataset.from_generator(parallel_gen,(tf.float64,tf.int64),(tf.TensorShape([ 16, 256, 9600]), tf.TensorShape([1,10])))
ds = ds.batch(2)

iter = ds.make_one_shot_iterator()
get_batch = iter.get_next()

with tf.Session() as sess:
   while True:
      x,y = sess.run(get_batch)
      print('x shape: %s; y shape: %s' % (x.shape,y.shape))
