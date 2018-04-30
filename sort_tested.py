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
	# print(df.columns.values)

	df1 = df[df.columns.values[2:len(df.columns.values)]]
	all_mins = df1.idxmin(1)

	# print(df['file_name'])

	for idx, file in enumerate(df['file_name']):
		# print(file)
		label = all_mins[idx]
		if not os.path.exists(FLAGS.out_dir+'/'+label):
			os.makedirs(FLAGS.out_dir+'/'+label)


		copyfile(file, FLAGS.out_dir+'/'+label+'/'+str(idx)+'.jpg')

	# for idx, row in df.iterrows():
	# 	print(row[2:len(df.columns.values)-2])
	# 	file_name = row['file_name']

	# 	# print(min(row[2:len(df.columns.values)-2]))
	



	# 	# for i in range(len(df.columns.values)-2):
	# 	# 	print(df.columns.values[i])
	


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

    FLAGS, unparsed = parser.parse_known_args()

    run_sorting()
