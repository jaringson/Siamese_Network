from __future__ import print_function
from __future__ import division

from pdb import set_trace as debugger
from tensorflow.python.training import moving_averages

import tensorflow as tf
import numpy as np

import os
import sys
import argparse

#import background
import glob
from os import path as osp
from PIL import Image
import pandas as pd


def run_testing():

    df = pd.read_csv(FLAGS.in_label)

    # print(df)

    sess = tf.Session()

    saver = tf.train.import_meta_graph(FLAGS.graph_path+'.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./tf_logs'))

    all_folders = glob.glob(FLAGS.testing_path+'/*')
    # all_test_files = glob.glob(osp.join(folder + FLAGS.load_run, '*'+ FLAGS.extension))

    raw_data = {'file_name':[]}

    

    for idx, row in df.iterrows():
        label = row['id']
        raw_data[str(label)] = []
    # print(train_file)
    # print(label)

    for folder in all_folders:
        
        if not osp.isdir(folder):
            continue
        for test_file in glob.glob(osp.join(folder + FLAGS.load_run, '*'+ FLAGS.extension)):
            raw_data['file_name'].append(test_file)
            for idx, row in df.iterrows():

                train_file = row['file_name']

                label = row['id']
                
                print(test_file)
                
                    
                train_img = Image.open(train_file).convert('L')
                train_img = train_img.resize((50,50), Image.ANTIALIAS)
                train_img = np.array(train_img)
                train_img = train_img / 255.0
                train_img = train_img.flatten().tolist()

                test_img = Image.open(test_file).convert('L')
                test_img = test_img.resize((50,50), Image.ANTIALIAS)
                test_img = np.array(test_img)
                test_img = test_img / 255.0
                test_img = test_img.flatten().tolist()

                
                feed2 = {'Placeholder:0': [train_img], 'Placeholder_1:0': [test_img], 
                        'Placeholder_2:0':True, 'Placeholder_3:0':2.25}
                en_out = sess.run(['energy/Sum:0'], feed_dict=feed2)
                #print(en_out[0][0][0])
                raw_data[str(label)].append(en_out[0][0][0])

                
    # print(raw_data)
    # print(raw_data.keys())
    all_keys = ['file_name']
    for i in range(len(raw_data.keys())-1):
        all_keys.append(str(i))
        # print(len(raw_data[i]))
    df = pd.DataFrame(raw_data, columns = all_keys)
    df.to_csv(osp.join(FLAGS.testing_path, FLAGS.result_file))

    sess.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
      '--in_label',
      type=str,
      default="./labels.csv",
      help='Number of training iterations.'
    )
    parser.add_argument(
      '--result_file',
      type=str,
      default='testing.csv',
      help='Results files of testings.'
    )
    parser.add_argument(
      '--graph_path',
      type=str,
      default='./tf_logs/siamese',
      help='Path to siamese graph.'
    )
    parser.add_argument(
      '--testing_path',
      type=str,
      default='./testing_data/',
      help='Path to testing data.'
    )
    parser.add_argument(
      '--load_run',
      type=str,
      default='/load1',
      help='Load run name.'
    )
    parser.add_argument(
      '--extension',
      type=str,
      default='.jpg',
      help='Extension (jpg, png, ect) of testing images.'
    )



    FLAGS, unparsed = parser.parse_known_args()

    run_testing()