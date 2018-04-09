#!/usr/bin/env python
'''from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

import re
import hashlib'''


from numpy import array
from PIL import Image
#import cv2
import glob
import os
from os import path as osp
import random
import math
#import count_white
#import imageProcessor2_0 as imp
import numpy as np
import paste_vary
from scipy.misc import imsave, imshow

import pandas as pd

#imgprocess = imp.TriangleWarp()

#image_dir = '/tf_files/modes/train_modes/'
#image_dir = './train_modes/'
#image_dir = './pratt_data_train/'


LABELS = {'file_name':[],'id':[]}

class BatchMaker(object):
	def __init__(self, image_dir):

		files = glob.glob(image_dir+'/*.jpg')


		for i,f in enumerate(files):
			#print os.listdir(os.path.join(image_dir, r))
			#temp = [0] * (len(sub_dirs) -1)
			#temp = [0] * 9 
			LABELS['id'].append(i) 
			LABELS['file_name'].append(f)

		if glob.glob('./lables.csv'):
			os.system("rm -f ./labels.csv")
		df = pd.DataFrame(LABELS, columns = ['file_name','id'])
		df.to_csv('./labels.csv')


	def next_batch(self, size, from_center =1, prev_labels =[]):
		#size = 10
		output = []
		output.append([])
		output.append([])
		output.append([])
		output.append([])
		output.append([])
		output.append([])

		

		for i in range(size):
			count_exepts = 0
			while True:
				label_num = -1
				if len(prev_labels) > 0:
					
					label_num = prev_labels[i]
					if np.random.randint(2):	
						while label_num == prev_labels[i]:
							label_num = random.randrange(len(LABELS))

				else:
					label_num = random.randrange(len(LABELS['file_name']))
				label = LABELS['id'][label_num]
				#image_num = random.randrange(LABELS[label]['tot_num'])
				image = LABELS['file_name'][label_num]
				try:
					thr_output = paste_vary.run(image)
					four_output = [x/255.0 for x in thr_output]
					img = Image.open(image)
					
					#output[0].append(list_of_lists)
					output[1].append([LABELS['id'][label]])
					#output[2].append(sec_output)
					output[3].append(thr_output)
					output[4].append(four_output)
					output[5].append(label_num)

				except:
					
					count_exepts = count_exepts + 1
					if count_exepts > 10:
						break
					continue
				break
				
			
		


		#print output
		return output

if __name__ == "__main__":
	bm = BatchMaker('./training_data/as0/load1')
	out = bm.next_batch(10,1)
	out2 = bm.next_batch(10,1,out[5])
	#print out
	#print out[4]
	print "-----------"
	print out[1] 
	print out2[1]
	print (np.array(out[1]) == np.array(out2[1])).astype(np.float32)
	print type(out[4])
	print len(out[4])
	print len(out[4][0])
	#r_img = np.array(out[3])
	#r_img = r_img.reshape([60,80])
	#imshow(r_img)
	
