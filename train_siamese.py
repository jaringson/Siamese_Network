###
# The code in this file alows for the actual training of the network.
# It uses 'siamese_network.py' to build and train the network and 
# 'utils.py' to create mini-batches for the creation of actual 
# training images.
###

from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

import os
from os import path as osp
import sys
import glob
import argparse

from utils import BatchMaker
from siamese_network import ModeSiameseNetwork

FLAGS = None

def run_training():

    s_net = ModeSiameseNetwork()

    ### prepare tf.session
    sess = tf.Session()

    l_summary = tf.summary.scalar('Loss', s_net.loss_op)
    acc_summary = tf.summary.scalar('Accuracy', s_net.accuracy_op)

    merged_summary_op = tf.summary.merge_all()

    if FLAGS.remove_prev and glob.glob("./tf_logs/events*"):
        os.system("rm -f ./tf_logs/events*")
    summary_writer = tf.summary.FileWriter("./tf_logs/", graph=sess.graph)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    if FLAGS.graph_path is not None:
        saver.restore(sess, FLAGS.graph_path)

    bm = BatchMaker(osp.join(FLAGS.training_data,FLAGS.load_run))
    for step in xrange(int(FLAGS.iterations)):
        
        batch_x1 = bm.next_batch(FLAGS.batch_size,1)
        batch_x2 = bm.next_batch(FLAGS.batch_size,1,batch_x1[5])
        batch_y = (np.array(batch_x1[1]) == np.array(batch_x2[1])).astype(np.float32)     

        feed = {s_net.x1: batch_x1[4], s_net.x2: batch_x2[4], 
                s_net.y_: batch_y, s_net.is_training: True, s_net.margin: 2.25}
        #feed2 = {x1: batch_x1[4], x2: batch_x2[4], is_training:True, margin:2.25}
        _, loss_v, acc_v, energy_v  = sess.run([s_net.train_step, s_net.loss_op,
                                         s_net.accuracy_op,s_net.energy_op], feed_dict=feed)
        #print(sess.run([energy_op], feed_dict=feed2))

        
        '''
        from scipy.misc import imsave
        for i in range(batch_size):
            img1 = np.array(batch_x1[4][i])
    	img1 = np.resize(img1,(50,50))
            img2 = np.array(batch_x2[4][i])
    	img2 = np.resize(img2,(50,50))
    	#print(img2.shape)
    	imsave('./tf_logs/'+out_d+'/'+str(i)+'_1.png',img1)
    	imsave('./tf_logs/'+out_d+'/'+str(i)+'_2.png',img2)
        '''

        
        if np.isnan(loss_v):
            print('Model diverged with loss = NaN')
            sys.exit()
        if step % 10 == 0:
            print('step {0}: loss {1} accuracy: {2}'.format(step, loss_v, acc_v))
        if step % 100 == 0:

            feed = {s_net.x1: batch_x1[4], s_net.x2: batch_x2[4],
                    s_net.y_: batch_y, s_net.is_training: True, s_net.margin: 2.25}
            summary_str, t_acc = sess.run([merged_summary_op,s_net.accuracy_op], feed_dict=feed)
            print('train set accuracy: {0}'.format(t_acc))
            summary_writer.add_summary(summary_str,step)
            saver.save(sess, "./tf_logs/siamese")

        sys.stdout.flush()
        

    saver.save(sess, "./tf_logs/siamese")
    summary_writer.close()
    sess.close()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
      '--iterations',
      type=str,
      default=3000,
      help='Number of training iterations.'
    )
    parser.add_argument(
      '--graph_path',
      type=str,
      default=None, #'./siamese',
      help='Path to siamese graph.'
    )
    parser.add_argument(
      '--batch_size',
      type=str,
      default=10,
      help='Size of mini-batch.'
    )
    parser.add_argument(
      '--training_data',
      type=str,
      default='./training_data',
      help='Path to folders of training images.'
    )
    parser.add_argument(
      '--load_run',
      type=str,
      default='as0/load1',
      help='Load run name.'
    )
    parser.add_argument(
      '--remove_prev',
      type=str,
      default=True,
      help='Remove previous runs.'
    )
    FLAGS, unparsed = parser.parse_known_args()

    run_training()
