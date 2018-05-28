###
# This script allows for the sorting of the testing files to
# different folders. It uses the output test_siamese to sort images.
#
###

from __future__ import print_function

import pandas as pd
import argparse
import numpy as np
import os
from shutil import copyfile
import ntpath



def run_sorting():

	if not os.path.exists(FLAGS.out_dir):
		os.makedirs(FLAGS.out_dir)

	df = pd.read_csv(FLAGS.in_file)

	for index, row in df.iterrows():
		file = row['file_name']
		min_val = float("inf")
		min_i = -1
		min_val2 = float("inf")
		min_i2 = -1
		for i in range(len(df.columns.values)-2):
			if row[str(i)] < min_val:
				min_i = i
				min_val = row[str(i)]
		for i in [x for x in xrange(len(df.columns.values)-2) if x != min_i]:
			if row[str(i)] < min_val2:
				min_i2 = i
				min_val2 = row[str(i)]
		

		if abs(min_val-min_val2) < FLAGS.threshold:
			if not os.path.exists(FLAGS.out_dir+'/99'):
				os.makedirs(FLAGS.out_dir+'/99')
			copyfile(file, FLAGS.out_dir+'/99/'+str(index)+'.jpg')
		else:
			if not os.path.exists(FLAGS.out_dir+'/'+str(min_i)):
				os.makedirs(FLAGS.out_dir+'/'+str(min_i))
			copyfile(file, FLAGS.out_dir+'/'+str(min_i)+'/'+str(index)+'.jpg')

	


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
      '--in_file',
      type=str,
      default="./testing_data/testing.csv",
      help='Infile name.'
    )

    parser.add_argument(
      '--out_dir',
      type=str,
      default="./output",
      help='Path to output directory.'
    )

    parser.add_argument(
      '--threshold',
      type=str,
      default=0.005,
      help='Threshold value.'
    )

    FLAGS, unparsed = parser.parse_known_args()

    run_sorting()
