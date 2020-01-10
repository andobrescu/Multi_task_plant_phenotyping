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


def get_data_A1A4(data_path, split_load):
	# Getting images (x data)
	imgname_train_A1 = np.array([glob.glob(data_path+'/CVPPP2017_LCC_training/TrainingSplits/A1'+str(h)+'/*.png') for h in split_load[0]])
	imgname_train_A4 = np.array([glob.glob(data_path+'/CVPPP2017_LCC_training/TrainingSplits/A4'+str(h)+'/*.png') for h in split_load[0]])
	imgname_val_A1 = np.array([glob.glob(data_path+'/CVPPP2017_LCC_training/TrainingSplits/A1'+str(split_load[1])+'/*.png')])
	imgname_val_A4 = np.array([glob.glob(data_path+'/CVPPP2017_LCC_training/TrainingSplits/A4'+str(split_load[1])+'/*.png')])
	imgname_test_A1 = np.array([glob.glob(data_path+'/CVPPP2017_LCC_training/TrainingSplits/A1'+str(split_load[2])+'/*.png')])
	imgname_test_A4 = np.array([glob.glob(data_path+'/CVPPP2017_LCC_training/TrainingSplits/A4'+str(split_load[2])+'/*.png')])

	filelist_train_A1 = list(np.sort(imgname_train_A1.flat)[1::2])
	filelist_train_A4 = list(np.sort(imgname_train_A4.flat)[1::2])

	filelist_train_A1_fg = list(np.sort(imgname_train_A1.flat)[0::2])
	filelist_train_A4_fg = list(np.sort(imgname_train_A4.flat)[0::2])

	filelist_train_A1_img = np.array([np.array(filelist_train_A1[h][-16:]) for h in range(0,len(filelist_train_A1))])
	filelist_train_A4_img = np.array([np.array(filelist_train_A4[h][-17:]) for h in range(0,len(filelist_train_A4))])

	filelist_train_A1_set = np.array([np.array(filelist_train_A1[h][-20:-18]) for h in range(0,len(filelist_train_A1))])
	filelist_train_A4_set = np.array([np.array(filelist_train_A4[h][-20:-18]) for h in range(0,len(filelist_train_A4))])

	filelist_val_A1 = list(np.sort(imgname_val_A1.flat)[1::2])
	filelist_val_A4 = list(np.sort(imgname_val_A4.flat)[1::2])

	filelist_val_A1_fg = list(np.sort(imgname_val_A1.flat)[0::2])
	filelist_val_A4_fg = list(np.sort(imgname_val_A4.flat)[0::2])

	filelist_val_A1_img = np.array([np.array(filelist_val_A1[h][-16:]) for h in range(0,len(filelist_val_A1))])
	filelist_val_A4_img = np.array([np.array(filelist_val_A4[h][-17:]) for h in range(0,len(filelist_val_A4))])

	filelist_val_A1_set = np.array([np.array(filelist_val_A1[h][-20:-18]) for h in range(0,len(filelist_val_A1))])
	filelist_val_A4_set = np.array([np.array(filelist_val_A4[h][-20:-18]) for h in range(0,len(filelist_val_A4))])

	filelist_test_A1 = list(np.sort(imgname_test_A1.flat)[1::2])
	filelist_test_A4 = list(np.sort(imgname_test_A4.flat)[1::2])

	filelist_test_A1_fg = list(np.sort(imgname_test_A1.flat)[0::2])
	filelist_test_A4_fg = list(np.sort(imgname_test_A4.flat)[0::2])

	filelist_test_A1_img = np.array([np.array(filelist_test_A1[h][-16:]) for h in range(0,len(filelist_test_A1))])
	filelist_test_A4_img = np.array([np.array(filelist_test_A4[h][-17:]) for h in range(0,len(filelist_test_A4))])
	filelist_test_A1_set = np.array([np.array(filelist_test_A1[h][-20:-18]) for h in range(0,len(filelist_test_A1))])
	filelist_test_A4_set = np.array([np.array(filelist_test_A4[h][-20:-18]) for h in range(0,len(filelist_test_A4))])


	x_train_A1 = np.array([np.array(Image.open(fname)) for fname in filelist_train_A1])
	x_train_A1 = np.delete(x_train_A1,3,3)
	x_train_A4 = np.array([np.array(Image.open(fname)) for fname in filelist_train_A4])

	x_train_A1_fg = np.array([np.array(Image.open(fname)) for fname in filelist_train_A1_fg])
	x_train_A4_fg = np.array([np.array(Image.open(fname)) for fname in filelist_train_A4_fg])


	x_val_A1 = np.array([np.array(Image.open(fname)) for fname in filelist_val_A1])
	x_val_A1 = np.delete(x_val_A1,3,3)
	x_val_A4 = np.array([np.array(Image.open(fname)) for fname in filelist_val_A4])

	x_val_A1_fg = np.array([np.array(Image.open(fname)) for fname in filelist_val_A1_fg])
	x_val_A4_fg = np.array([np.array(Image.open(fname)) for fname in filelist_val_A4_fg])

	x_test_A1 = np.array([np.array(Image.open(fname)) for fname in filelist_test_A1])
	x_test_A1 = np.delete(x_test_A1,3,3)
	x_test_A4 = np.array([np.array(Image.open(fname)) for fname in filelist_test_A4])

	x_test_A1_fg = np.array([np.array(Image.open(fname)) for fname in filelist_test_A1_fg])
	x_test_A4_fg = np.array([np.array(Image.open(fname)) for fname in filelist_test_A4_fg])

	x_train_res_A1 = np.array([misc.imresize(x_train_A1[i],[317,309,3]) for i in range(0,len(x_train_A1))])
	x_train_res_A4 = np.array([misc.imresize(x_train_A4[i],[317,309,3]) for i in range(0,len(x_train_A4))])

	x_val_res_A1 = np.array([misc.imresize(x_val_A1[i],[317,309,3]) for i in range(0,len(x_val_A1))])
	x_val_res_A4 = np.array([misc.imresize(x_val_A4[i],[317,309,3]) for i in range(0,len(x_val_A4))])

	x_test_res_A1 = np.array([misc.imresize(x_test_A1[i],[317,309,3]) for i in range(0,len(x_test_A1))])
	x_test_res_A4 = np.array([misc.imresize(x_test_A4[i],[317,309,3]) for i in range(0,len(x_test_A4))])

	x_train_res_A1_fg = np.array([misc.imresize(x_train_A1_fg[i],[317,309,3]) for i in range(0,len(x_train_A1_fg))])
	x_train_res_A4_fg = np.array([misc.imresize(x_train_A4_fg[i],[317,309,3]) for i in range(0,len(x_train_A4_fg))])

	x_val_res_A1_fg = np.array([misc.imresize(x_val_A1_fg[i],[317,309,3]) for i in range(0,len(x_val_A1))])
	x_val_res_A4_fg = np.array([misc.imresize(x_val_A4_fg[i],[317,309,3]) for i in range(0,len(x_val_A4))])

	x_test_res_A1_fg = np.array([misc.imresize(x_test_A1_fg[i],[317,309,3]) for i in range(0,len(x_test_A1_fg))])
	x_test_res_A4_fg = np.array([misc.imresize(x_test_A4_fg[i],[317,309,3]) for i in range(0,len(x_test_A4_fg))])

	x_train_all = np.concatenate((x_train_res_A1, x_train_res_A4), axis=0)
	x_val_all = np.concatenate((x_val_res_A1, x_val_res_A4), axis=0)
	x_test_all = np.concatenate((x_test_res_A1, x_test_res_A4), axis=0)

	for h in range(0,len(x_train_all)):
		x_img = x_train_all[h]
		x_img_pil = Image.fromarray(x_img)
		x_img_pil = ImageOps.autocontrast(x_img_pil)
		x_img_ar = np.array(x_img_pil)
		x_train_all[h] = x_img_ar

	for h in range(0,len(x_val_all)):
		x_img = x_val_all[h]
		x_img_pil = Image.fromarray(x_img)
		x_img_pil = ImageOps.autocontrast(x_img_pil)
		x_img_ar = np.array(x_img_pil)
		x_val_all[h] = x_img_ar

	for h in range(0,len(x_test_all)):
		x_img = x_test_all[h]
		x_img_pil = Image.fromarray(x_img)
		x_img_pil = ImageOps.autocontrast(x_img_pil)
		x_img_ar = np.array(x_img_pil)
		x_test_all[h] = x_img_ar

	x_train_all_fg = np.concatenate((x_train_res_A1_fg, x_train_res_A4_fg), axis=0)
	x_val_all_fg  = np.concatenate((x_val_res_A1_fg, x_val_res_A4_fg), axis=0)
	x_test_all_fg = np.concatenate((x_test_res_A1_fg, x_test_res_A4_fg), axis=0)

	sum_train_all = np.zeros((len(x_train_all_fg),1))
	sum_val_all = np.zeros((len(x_val_all_fg),1))
	sum_test_all = np.zeros((len(x_test_all_fg),1))

	for i in range(0, len(x_train_all_fg)):
		x_train_all_fg[i][x_train_all_fg[i] > 0] = 1
		sum_train_all[i] = np.sum(x_train_all_fg[i])

	for i in range(0, len(x_val_all_fg)):
		x_val_all_fg[i][x_val_all_fg[i] > 0] = 1
		sum_val_all[i] = np.sum(x_val_all_fg[i])

	for i in range(0, len(x_test_all_fg)):
		x_test_all_fg[i][x_test_all_fg[i] > 0] = 1
		sum_test_all[i] = np.sum(x_test_all_fg[i])


	x_train_img = np.concatenate((filelist_train_A1_img, filelist_train_A4_img), axis=0)
	x_val_img = np.concatenate((filelist_val_A1_img, filelist_val_A4_img), axis=0)
	x_test_img = np.concatenate((filelist_test_A1_img, filelist_test_A4_img), axis=0)

	x_train_set = np.concatenate((filelist_train_A1_set, filelist_train_A4_set), axis=0)
	x_val_set = np.concatenate((filelist_val_A1_set, filelist_val_A4_set), axis=0)
	x_test_set = np.concatenate((filelist_test_A1_set, filelist_test_A4_set), axis=0)
	

	# Getting targets (y data)	  #
	counts_A1 = np.array([glob.glob(data_path+'/CVPPP2017_LCC_training/TrainingSplits/A1.xlsx')])
	counts_A4 = np.array([glob.glob(data_path+'/CVPPP2017_LCC_training/TrainingSplits/A4.xlsx')])


	counts_train_flat_A1 = list(counts_A1.flat)
	train_labels_A1 = pd.DataFrame()
	y_train_A1_list = []
	y_val_A1_list = []
	y_test_A1_list = []
	for f in counts_train_flat_A1:
		frame = pd.read_excel(f, header=None)
		train_labels_A1 = train_labels_A1.append(frame, ignore_index=False)
	all_labels_A1 = np.array(train_labels_A1)

	for j in filelist_train_A1_img:
		arr_idx = np.where(all_labels_A1 == j)
		y_train_A1_list.append(all_labels_A1[arr_idx[0],:])
	y_train_A1_labels = np.concatenate(y_train_A1_list, axis=0)

	for j in filelist_val_A1_img:
		arr_idx = np.where(all_labels_A1 == j)
		y_val_A1_list.append(all_labels_A1[arr_idx[0],:])
	y_val_A1_labels = np.concatenate(y_val_A1_list, axis=0)

	for j in filelist_test_A1_img:
		arr_idx = np.where(all_labels_A1 == j)
		y_test_A1_list.append(all_labels_A1[arr_idx[0],:])
	y_test_A1_labels = np.concatenate(y_test_A1_list, axis=0)


	counts_train_flat_A4 = list(counts_A4.flat)
	train_labels_A4 = pd.DataFrame()
	y_train_A4_list = []
	y_val_A4_list = []
	y_test_A4_list = []
	for f in counts_train_flat_A4:
		frame = pd.read_excel(f, header=None)
		train_labels_A4 = train_labels_A4.append(frame, ignore_index=False)
	all_labels_A4 = np.array(train_labels_A4)

	for j in filelist_train_A4_img:
		arr_idx = np.where(all_labels_A4 == j)
		y_train_A4_list.append(all_labels_A4[arr_idx[0],:])
	y_train_A4_labels = np.concatenate(y_train_A4_list, axis=0)

	for j in filelist_val_A4_img:
		arr_idx = np.where(all_labels_A4 == j)
		y_val_A4_list.append(all_labels_A4[arr_idx[0],:])
	y_val_A4_labels = np.concatenate(y_val_A4_list, axis=0)

	for j in filelist_test_A4_img:
		arr_idx = np.where(all_labels_A4 == j)
		y_test_A4_list.append(all_labels_A4[arr_idx[0],:])
	y_test_A4_labels = np.concatenate(y_test_A4_list, axis=0)


	y_train_all_labels = np.concatenate((y_train_A1_labels, y_train_A4_labels), axis=0)
	y_val_all_labels = np.concatenate((y_val_A1_labels, y_val_A4_labels), axis=0)
	y_test_all_labels = np.concatenate((y_test_A1_labels, y_test_A4_labels), axis=0)

	y_train_all = y_train_all_labels[:,1]
	y_val_all = y_val_all_labels[:,1]
	y_test_all = y_test_all_labels[:,1]

	return x_train_all, x_val_all, x_test_all, y_train_all, y_val_all, y_test_all, x_train_set, x_val_set, x_test_set, x_train_img, x_val_img, x_test_img, x_train_all_fg, x_val_all_fg, x_test_all_fg, sum_train_all, sum_val_all, sum_test_all

