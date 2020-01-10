import os
import traceback
import time
import scipy.misc as misc
import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
import random
import cv2
import h5py
from PIL import Image, ImageOps
from datetime import datetime
from collections import OrderedDict

import keras
import keras.backend as K
from keras import callbacks
from keras.activations import sigmoid
from keras.models import Sequential, Model, load_model
from keras.utils import to_categorical

from data.Load_data_pheno import load_data_mixed
from data.Load_data_CVPPP_A1A4 import get_data_A1A4
from data.Load_data_CVPPP_testing import get_data_testing
from model.model_multitask import multi_task_model_mixed , counter_model_augmentation
from compute_statistics_multi import testing_results

print('Welcome to my program')
# At the moment the program requires to load the datasets. If using on different datasets or
# there is no access to the datasets used in the paper delete the lines 36-88. 
# ToDo make dataset import dynamic

datasets_path = '/home/andrei/Datasets' # Change to datasets path


# Load phenotiki dataset
data_pheno = load_data_mixed(datasets_path)
x_train_count = data_pheno[0]
x_train_gen = data_pheno[1]
x_test_all = data_pheno[2]
y_count = data_pheno[3]
y_genotype = data_pheno[4]
y_PLA = data_pheno[5]
sets = data_pheno[6]
img_sets = data_pheno[7]
print('Done loading the Phenotiki Dataset')

# split if doing cross validation
split_load = [[2, 3], 4, 1]

# Load cvppp dataset
data_cvppp = get_data_A1A4(datasets_path, split_load)
x_train_cvppp = data_cvppp[0]
x_val_cvppp = data_cvppp[1]
x_test_cvppp = data_cvppp[2]
y_train_cvppp = data_cvppp[3]
y_val_cvppp = data_cvppp[4]
y_test_cvppp = data_cvppp[5]
x_train_cvppp_fg = data_cvppp[12]
x_val_cvppp_fg = data_cvppp[13]
x_test_cvppp_fg = data_cvppp[14]

PLA_training_cvppp = []
PLA_val_cvppp = []
PLA_test_cvppp = []

step = 0
total_img_area = 317 * 309

PLA_training_pheno = y_PLA[0] / total_img_area
PLA_test_pheno = y_PLA[1] / total_img_area
PLA_train_pheno_gen = y_PLA[2] / total_img_area

for i in range(len(x_train_cvppp_fg)):
	PLA_training_cvppp.append((np.sum(x_train_cvppp_fg[i]) / total_img_area))
for i in range(len(x_val_cvppp_fg)):
	PLA_val_cvppp.append((np.sum(x_val_cvppp_fg[i] / total_img_area)))
for i in range(len(x_test_cvppp_fg)):
	PLA_test_cvppp.append((np.sum(x_test_cvppp_fg[i] / total_img_area)))

PLA_training_cvppp = np.array(PLA_training_cvppp)
PLA_val_cvppp = np.array(PLA_val_cvppp)
PLA_test_cvppp = np.array(PLA_test_cvppp)

y_cvppp_gen_train = np.zeros([len(x_train_cvppp)])
y_cvppp_gen_test = np.zeros([len(x_test_cvppp)])

print('Done loading the CVPPP Dataset')


# data_cvppp_testing = get_data_testing()
# x_testing_set_cvppp_A1 = data_cvppp_testing[0]
# x_testing_set_cvppp_A4 = data_cvppp_testing[3]
# y_testing_set_cvppp_A1 = data_cvppp_testing[5]
# y_testing_set_cvppp_A4 = data_cvppp_testing[8]


mask_value = -1

# Compile a training dataset:
# 'pheno' is the phenotiki dataset as seen in Minervini 2017: Phenotiki: an open software and hardware
#  platform for affordable and easy image-based phenotyping of rosette-shaped plants
# 'pheno_gen' is the phenotiki dataset mentioned above but unlabeled images wild-type Col-O are added
#  so that there is balance between wild-type and mutant plant images
# 'cvppp' is the 2017 computer vision problems in plant phenotyping,leaf counting challenge dataset
# 'cvppp_testing_A1' the test set from the cvppp leaf counting challenge
# 'pheno+cvppp' a combination of the cvppp and pheno datasets
# 'pheno_gen+cvppp' a combination of pheno_gen and the cvppp datasets

def process_data(dataset='pheno'):
	training_data = []

	if dataset == 'pheno':

		training_data.append(x_train_count)
		training_data.append(x_test_all)
		training_data.append(y_count[0])
		training_data.append(y_count[1])
		training_data.append(y_genotype[0])
		training_data.append(y_genotype[1])
		training_data.append(PLA_training_pheno)
		training_data.append(PLA_test_pheno)

	elif dataset == 'pheno_gen':

		training_data.append(x_train_gen)
		training_data.append(x_test_all)
		training_data.append(y_count[2])
		training_data.append(y_count[1])
		training_data.append(y_genotype[2])
		training_data.append(y_genotype[1])
		training_data.append(PLA_train_pheno_gen)
		training_data.append(PLA_test_pheno)

	elif dataset == 'cvppp':

		training_data.append(x_train_cvppp)
		training_data.append(x_test_cvppp)
		training_data.append(y_train_cvppp)
		training_data.append(y_test_cvppp)
		training_data.append(y_cvppp_gen_train)
		training_data.append(y_cvppp_gen_test)
		training_data.append(PLA_training_cvppp)
		training_data.append(PLA_test_cvppp)

	elif dataset == 'cvppp_testing_A1':

		training_data.append(x_train_cvppp)
		training_data.append(x_testing_set_cvppp_A1)
		training_data.append(y_train_cvppp)
		training_data.append(y_testing_set_cvppp_A1)

	elif dataset == 'cvppp_testing_A4':

		training_data.append(x_train_cvppp)
		training_data.append(x_testing_set_cvppp_A4)
		training_data.append(y_train_cvppp)
		training_data.append(y_testing_set_cvppp_A4)


	elif dataset == 'pheno+cvppp':

		# Mix Pheno data with CVPPP
		x_train_mix = np.concatenate([x_train_count, x_train_cvppp], axis=0)
		x_test_mix = np.concatenate([x_test_all, x_test_cvppp], axis=0)
		y_train_count_mix = np.concatenate([y_count[0], y_train_cvppp], axis=0)
		y_test_count_mix = np.concatenate([y_count[1], y_test_cvppp], axis=0)
		y_train_gen_mix = np.concatenate([y_genotype[0], y_cvppp_gen_train], axis=0)
		y_test_gen_mix = np.concatenate([y_genotype[1], y_cvppp_gen_test], axis=0)
		y_train_pla_mix = np.concatenate([PLA_training_pheno, PLA_training_cvppp], axis=0)
		y_test_pla_mix = np.concatenate([PLA_test_pheno, PLA_test_cvppp], axis=0)


		training_data.append(x_train_mix)
		training_data.append(x_test_mix)
		training_data.append(y_train_count_mix)
		training_data.append(y_test_count_mix)
		training_data.append(y_train_gen_mix)
		training_data.append(y_test_gen_mix)
		training_data.append(y_train_pla_mix)
		training_data.append(y_test_pla_mix)

	elif dataset == 'pheno_gen+cvppp':

		# Mix Pheno data with CVPPP
		x_train_mix = np.concatenate([x_train_gen, x_train_cvppp], axis=0)
		x_test_mix = np.concatenate([x_test_all, x_test_cvppp], axis=0)
		y_train_count_mix = np.concatenate([y_count[2], y_train_cvppp], axis=0)
		y_test_count_mix = np.concatenate([y_count[1], y_test_cvppp], axis=0)
		y_train_gen_mix = np.concatenate([y_genotype[2], y_cvppp_gen_train], axis=0)
		y_test_gen_mix = np.concatenate([y_genotype[1], y_cvppp_gen_test], axis=0)
		y_train_pla_mix = np.concatenate([PLA_train_pheno_gen, PLA_training_cvppp], axis=0)
		y_test_pla_mix = np.concatenate([PLA_test_pheno, PLA_test_cvppp], axis=0)


		training_data.append(x_train_mix)
		training_data.append(x_test_mix)
		training_data.append(y_train_count_mix)
		training_data.append(y_test_count_mix)
		training_data.append(y_train_gen_mix)
		training_data.append(y_test_gen_mix)
		training_data.append(y_train_pla_mix)
		training_data.append(y_test_pla_mix)

	else:
		print('Need a valid dataset specification!')

	return training_data




def PLA_loss(y_true, y_pred):
	return K.mean(K.square((y_pred - y_true) * 10), axis=-1)





def MSE_masked_loss(y_true, y_pred):
	mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
	return K.mean(K.square(y_pred * mask - y_true * mask), axis=-1)

def mse_discrete_accuracy(y_true, y_pred):
	return K.mean(K.square(K.round(y_pred) - y_true), axis=-1)

def train_the_model_multi(results_path, data, miss_labels, gen_weight):
	trained_model = multi_task_model_mixed(results_path, data, miss_labels, gen_weight)
	return trained_model

def train_the_model_single(results_path, data, miss_labels):
	trained_model = counter_model_augmentation(results_path, data, miss_labels)
	return trained_model

# Define what datasets you want to use for training as a list 
dataset_multi = ['pheno']

# Define the percentage of missing labels per dataset
labels_percent_multi = [0.50]

# Define the weight for the wild-type image if the dataset is unbalanced such as in the pheno dataset
col_weight = [5.0]

# The values above can be a list to train multiple models in a sequence: 
# e.g dataset_multi = ['pheno', 'cvppp', 'pheno_gen+cvpp']   


try:
	for i in range(len(dataset_multi)):
		current_time = datetime.now().strftime('%m-%d %H:%M')
		results_path = ('./Results/MTL_results '+current_time)
		os.makedirs(results_path)

		missing_labels = labels_percent_multi[i]
		feed_data = process_data(dataset_multi[i])

		set_text = open(results_path +'/sets.txt', 'w+')
		set_text.write('Train set is '+ dataset_multi[i] + ' \r\n')
		set_text.write('Training plants are ' +str(sets[0])+ ' \r\n')
		set_text.write('Test set is '+ str(sets[1])+ ' \r\n')
		# set_text.write('train gen set is '+ str(sets[2])+ ' \r\n')
		set_text.write('Percent Missing labels is  '+ str(missing_labels)+ ' \r\n')
		set_text.write('Stats for cvppp'+ str(split_load))
		set_text.close()

		
		trained_model = train_the_model_multi(results_path, feed_data, missing_labels, col_weight[i])
		# check_path = 'Results/TwoTaskCvppp 02-13 11:58'
		# time.sleep(60)

		# trained_model = load_model(check_path+'/checkpoint.hdf5', custom_objects= {'MSE_masked_loss' : MSE_masked_loss,
		# 																	'mse_discrete_accuracy': mse_discrete_accuracy})

		results_stats = testing_results_multi(trained_model, feed_data, results_path)
		
		del trained_model

except Exception:
	traceback.print_exc()




