from __future__ import division, print_function, absolute_import

import os
import traceback
import scipy.misc as misc
import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
import random
from PIL import Image, ImageOps


def get_data_testing(data_path):
	# Getting test images (x data)
	imgname_test_A1 = np.array([glob.glob(data_path+'/CVPPP2017_testing/testing/A1/*rgb.png')])
	imgname_test_A2 = np.array([glob.glob(data_path+'/CVPPP2017_testing/testing/A2/*rgb.png')])
	imgname_test_A3 = np.array([glob.glob(data_path+'/CVPPP2017_testing/testing/A3/*rgb.png')])
	imgname_test_A4 = np.array([glob.glob(data_path+'/CVPPP2017_testing/testing/A4/*rgb.png')])
	imgname_test_A5 = np.array([glob.glob(data_path+'/CVPPP2017_testing/testing/A5/*rgb.png')])


	filelist_test_A1 = list(np.sort(imgname_test_A1.flat))
	filelist_test_A2 = list(np.sort(imgname_test_A2.flat))
	filelist_test_A3 = list(np.sort(imgname_test_A3.flat))
	filelist_test_A4 = list(np.sort(imgname_test_A4.flat))
	filelist_test_A5 = list(np.sort(imgname_test_A5.flat))

	filelist_test_A1_img = np.array([np.array(filelist_test_A1[h][-16:]) for h in range(0,len(filelist_test_A1))])
	filelist_test_A2_img = np.array([np.array(filelist_test_A2[h][-16:]) for h in range(0,len(filelist_test_A2))])
	filelist_test_A3_img = np.array([np.array(filelist_test_A3[h][-16:]) for h in range(0,len(filelist_test_A3))])
	filelist_test_A4_img = np.array([np.array(filelist_test_A4[h][-18:]) for h in range(0,len(filelist_test_A4))])
	filelist_test_A5_img = np.array([np.array(filelist_test_A5[h][-18:]) for h in range(0,len(filelist_test_A5))])

	x_test_A1 = np.array([np.array(misc.imread(fname)) for fname in filelist_test_A1])
	x_test_A1 = np.delete(x_test_A1,3,3)
	x_test_A2 = np.array([np.array(Image.open(fname)) for fname in filelist_test_A2])
	x_test_A2 = np.delete(x_test_A2,3,3)
	x_test_A3 = np.array([np.array(Image.open(fname)) for fname in filelist_test_A3])
	x_test_A4 = np.array([np.array(Image.open(fname)) for fname in filelist_test_A4])
	x_test_A5 = np.array([np.array(Image.open(fname)) for fname in filelist_test_A5])

	for i in range(0, len(x_test_A5)):
		x_A5_img = x_test_A5[i]
		if x_A5_img.shape[2] == 4:
			x_A5_img_del = np.delete(x_A5_img,3,2)
			x_test_A5[i] = x_A5_img_del

	x_test_res_A1 = np.array([misc.imresize(x_test_A1[i],[317,309,3]) for i in range(0,len(x_test_A1))])
	x_test_res_A2 = np.array([misc.imresize(x_test_A2[i],[317,309,3]) for i in range(0,len(x_test_A2))])
	x_test_res_A3 = np.array([misc.imresize(x_test_A3[i],[317,309,3]) for i in range(0,len(x_test_A3))])
	x_test_res_A4 = np.array([misc.imresize(x_test_A4[i],[317,309,3]) for i in range(0,len(x_test_A4))])
	x_test_res_A5 = np.array([misc.imresize(x_test_A5[i],[317,309,3]) for i in range(0,len(x_test_A5))])

	for h in range(0,len(x_test_res_A1)):
		x_img = x_test_res_A1[h]
		x_img_pil = Image.fromarray(x_img)
		x_img_pil = ImageOps.autocontrast(x_img_pil)
		x_img_ar = np.array(x_img_pil)
		x_test_res_A1[h] = x_img_ar

	for h in range(0,len(x_test_res_A2)):
		x_img = x_test_res_A2[h]
		x_img_pil = Image.fromarray(x_img)
		x_img_pil = ImageOps.autocontrast(x_img_pil)
		x_img_ar = np.array(x_img_pil)
		x_test_res_A2[h] = x_img_ar

	for h in range(0,len(x_test_res_A3)):
		x_img = x_test_res_A3[h]
		x_img_pil = Image.fromarray(x_img)
		x_img_pil = ImageOps.autocontrast(x_img_pil)
		x_img_ar = np.array(x_img_pil)
		x_test_res_A3[h] = x_img_ar

	for h in range(0,len(x_test_res_A4)):
		x_img = x_test_res_A4[h]
		x_img_pil = Image.fromarray(x_img)
		x_img_pil = ImageOps.autocontrast(x_img_pil)
		x_img_ar = np.array(x_img_pil)
		x_test_res_A4[h] = x_img_ar
	
	for h in range(0,len(x_test_res_A5)):
		x_img = x_test_res_A5[h]
		x_img_pil = Image.fromarray(x_img)
		x_img_pil = ImageOps.autocontrast(x_img_pil)
		x_img_ar = np.array(x_img_pil)
		x_test_res_A5[h] = x_img_ar

	# Get GT values
	file_y_A1 = pd.read_csv(data_path+'/CVPPP2017_testing/testing/A1.csv', header=None)
	file_y_A2 = pd.read_csv(data_path+'/CVPPP2017_testing/testing/A2.csv', header=None)
	file_y_A3 = pd.read_csv(data_path+'/CVPPP2017_testing/testing/A3.csv', header=None)
	file_y_A4 = pd.read_csv(data_path+'/CVPPP2017_testing/testing/A4.csv', header=None)
	file_y_A5 = pd.read_csv(data_path+'/CVPPP2017_testing/testing/A5.csv', header=None)

	y_test_A1 = file_y_A1.as_matrix()[:,1]
	y_test_A2 = file_y_A2.as_matrix()[:,1]
	y_test_A3 = file_y_A3.as_matrix()[:,1]
	y_test_A4 = file_y_A4.as_matrix()[:,1]
	y_test_A5 = file_y_A5.as_matrix()[:,1]


	return x_test_res_A1, x_test_res_A2, x_test_res_A3, x_test_res_A4, x_test_res_A5, y_test_A1, y_test_A2, y_test_A3, y_test_A4, y_test_A5



