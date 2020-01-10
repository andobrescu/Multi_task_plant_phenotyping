import os
import traceback
import scipy.misc as misc
import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
import random
import h5py
from PIL import Image, ImageOps
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime
from collections import OrderedDict

import keras
import keras.backend as K
from keras import regularizers
from keras import callbacks
from keras.activations import sigmoid
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, merge, ConvLSTM2D, Reshape
from keras.layers import Input, Convolution2D, MaxPooling2D, LeakyReLU, LSTM, TimeDistributed, Conv2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import metrics
from keras import backend as K
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical


def testing_results(model, data, results_path):
	
	x_train_all = data[0]
	x_test_all = data[1]
	y_train_count = data[2]
	y_test_count = data[3]
	y_train_genotype = data[4]
	y_test_genotype = data[5]
	y_train_PLA = data[6]
	y_test_PLA = data[7]

	res_total_size = len(y_train_count)
	res_test_size = len(y_test_count)
	print('Calculating Statistics')
	print('Training Statistics')
	# Training set results #
	predictions_train = model.predict(x_train_all)
	predictions_train_count = predictions_train[0]
	predictions_train_PLA = predictions_train[1]
	predictions_train_genotype = predictions_train[2]

	y_train_genotype = y_train_genotype
	y_test_genotype = y_test_genotype
	
	# Count
	pred_count_round_train = np.round(predictions_train_count)
	pred_count_round_train = np.array(pred_count_round_train, dtype= int)
	y_train_count = np.reshape(y_train_count, (len(y_train_count),1))
	res_train_count = np.concatenate((y_train_count, pred_count_round_train) , axis =1)

	difference_count_train = np.array(np.round([(res_train_count[h,1]-res_train_count[h,0]) for h in range(0,len(y_train_count))]))
	difference_count_train = difference_count_train.reshape(difference_count_train.size, 1)
	difference_std_count_train = np.std(difference_count_train)
	average_diff_train_count = np.average(difference_count_train)

	difference_count_abs_train = np.array(np.round([abs(res_train_count[h,1]-res_train_count[h,0]) for h in range(0,len(y_train_count))]))
	difference_count_abs_train = difference_count_abs_train.reshape(difference_count_abs_train.size, 1)
	difference_std_abs_count_train = np.std(difference_count_abs_train)
	average_diff_abs_train_count = np.average(difference_count_abs_train)

	prediction_equal_train = np.equal(res_train_count[:,0],res_train_count[:,1])
	prediction_equal_train = prediction_equal_train.astype(int)
	prediction_equal_train = np.reshape(prediction_equal_train, (len(res_train_count),1))

	res_train_count = np.concatenate((res_train_count,prediction_equal_train), axis=1)

	mean_arr_train = np.mean(res_train_count[:,1], axis=0)
	r_coeff_train = r2_score(y_train_count, pred_count_round_train)
	MSE_train = mean_squared_error(y_train_count,pred_count_round_train)
	agreement_sum_train = np.sum(prediction_equal_train)
	agreement_train = agreement_sum_train/len(y_train_count)

	#Age
	pred_PLA_round_train = np.round(predictions_train_PLA)
	pred_PLA_round_train = np.array(pred_PLA_round_train, dtype= int)
	y_train_PLA = np.reshape(y_train_PLA, (len(y_train_PLA),1))
	res_train_PLA = np.concatenate((y_train_PLA, pred_PLA_round_train) , axis =1)

	difference_PLA_train = np.array(np.round([(res_train_PLA[h,1]-res_train_PLA[h,0]) for h in range(0,len(y_train_PLA))]))
	difference_PLA_train = difference_PLA_train.reshape(difference_PLA_train.size, 1)
	difference_std_PLA_train = np.std(difference_PLA_train)
	average_diff_train_PLA = np.average(difference_PLA_train)

	difference_PLA_abs_train = np.array(np.round([abs(res_train_PLA[h,1]-res_train_PLA[h,0]) for h in range(0,len(y_train_PLA))]))
	difference_PLA_abs_train = difference_PLA_abs_train.reshape(difference_PLA_abs_train.size, 1)
	difference_std_abs_PLA_train = np.std(difference_PLA_abs_train)
	average_diff_abs_train_PLA = np.average(difference_PLA_abs_train)

	prediction_equal_train_PLA = np.equal(res_train_PLA[:,0],res_train_PLA[:,1])
	prediction_equal_train_PLA = prediction_equal_train_PLA.astype(int)
	prediction_equal_train_PLA = np.reshape(prediction_equal_train_PLA, (len(res_train_PLA),1))

	res_train_PLA = np.concatenate((res_train_PLA,prediction_equal_train_PLA), axis=1)

	mean_arr_train_PLA = np.mean(res_train_PLA[:,1], axis=0)
	r_coeff_train_PLA = r2_score(y_train_PLA, pred_PLA_round_train)
	MSE_train_PLA = mean_squared_error(y_train_PLA,pred_PLA_round_train)
	agreement_sum_train_PLA = np.sum(prediction_equal_train_PLA)
	agreement_train_PLA = agreement_sum_train_PLA/len(y_train_PLA)

	#Genotype
	pred_genotype_train = np.empty([res_total_size,1], dtype = int)
	for i in range(len(predictions_train_genotype)):
		pred_genotype_train[i] = np.argmax(predictions_train_genotype[i])
	print(y_train_genotype.shape)

	train_genotype_train = np.empty([res_total_size, 1], dtype=int)
	for i in range(len(y_train_genotype)):
		train_genotype_train[i] = np.argmax(y_train_genotype[i])
	y_train_genotype = np.copy(train_genotype_train)
	# y_train_genotype = np.reshape(y_train_genotype, (len(y_train_genotype),1))
	res_train_genotype = np.concatenate((y_train_genotype, pred_genotype_train), axis=1)

	prediction_equal_train_genotype = np.equal(res_train_genotype[:,0],res_train_genotype[:, 1])
	prediction_equal_train_genotype = prediction_equal_train_genotype.astype(int)
	prediction_equal_train_genotype = np.reshape(prediction_equal_train_genotype, (len(res_train_genotype),1))

	res_train_genotype = np.concatenate((res_train_genotype,prediction_equal_train_genotype), axis=1)

	agreement_sum_train_genotype = np.sum(prediction_equal_train_genotype)
	agreement_train_genotype = agreement_sum_train_genotype/len(y_train_genotype)

	# Test set results #
	print('Test Statistics')
	predictions_test = model.predict(x_test_all)
	predictions_test_count = predictions_test[0]
	predictions_test_PLA = predictions_test[1]
	predictions_test_genotype = predictions_test[2]

	pred_count_round_test = np.round(predictions_test_count)
	pred_count_round_test = np.array(pred_count_round_test, dtype= int)
	y_test_count = np.reshape(y_test_count, (len(y_test_count),1))
	res_test_count = np.concatenate((y_test_count, pred_count_round_test) , axis =1)

	difference_count_test = np.array(np.round([(res_test_count[h,1]-res_test_count[h,0]) for h in range(0,len(y_test_count))]))
	difference_count_test = difference_count_test.reshape(difference_count_test.size, 1)
	difference_std_count_test = np.std(difference_count_test)
	average_diff_test_count = np.average(difference_count_test)

	difference_count_abs_test = np.array(np.round([abs(res_test_count[h,1]-res_test_count[h,0]) for h in range(0,len(y_test_count))]))
	difference_count_abs_test = difference_count_abs_test.reshape(difference_count_abs_test.size, 1)
	difference_std_abs_count_test = np.std(difference_count_abs_test)
	average_diff_abs_test_count = np.average(difference_count_abs_test)

	prediction_equal_test = np.equal(res_test_count[:,0],res_test_count[:,1])
	prediction_equal_test = prediction_equal_test.astype(int)
	prediction_equal_test = np.reshape(prediction_equal_test, (len(res_test_count),1))

	res_test_count = np.concatenate((res_test_count,prediction_equal_test), axis=1)

	mean_arr_test = np.mean(res_test_count[:,1], axis=0)
	r_coeff_test = r2_score(y_test_count, pred_count_round_test)
	MSE_test = mean_squared_error(y_test_count,pred_count_round_test)
	agreement_sum_test = np.sum(prediction_equal_test)
	agreement_test = agreement_sum_test/len(y_test_count)

	#Age
	pred_PLA_round_test = np.round(predictions_test_PLA)
	pred_PLA_round_test = np.array(pred_PLA_round_test, dtype= int)
	y_test_PLA = np.reshape(y_test_PLA, (len(y_test_PLA),1))
	res_test_PLA = np.concatenate((y_test_PLA, pred_PLA_round_test) , axis =1)

	difference_PLA_test = np.array(np.round([(res_test_PLA[h,1]-res_test_PLA[h,0]) for h in range(0,len(y_test_PLA))]))
	difference_PLA_test = difference_PLA_test.reshape(difference_PLA_test.size, 1)
	difference_std_PLA_test = np.std(difference_PLA_test)
	average_diff_test_PLA = np.average(difference_PLA_test)

	difference_PLA_abs_test = np.array(np.round([abs(res_test_PLA[h,1]-res_test_PLA[h,0]) for h in range(0,len(y_test_PLA))]))
	difference_PLA_abs_test = difference_PLA_abs_test.reshape(difference_PLA_abs_test.size, 1)
	difference_std_abs_PLA_test = np.std(difference_PLA_abs_test)
	average_diff_abs_test_PLA = np.average(difference_PLA_abs_test)

	prediction_equal_test_PLA = np.equal(res_test_PLA[:,0],res_test_PLA[:,1])
	prediction_equal_test_PLA = prediction_equal_test_PLA.astype(int)
	prediction_equal_test_PLA = np.reshape(prediction_equal_test_PLA, (len(res_test_PLA),1))

	res_test_PLA = np.concatenate((res_test_PLA,prediction_equal_test_PLA), axis=1)

	mean_arr_test_PLA = np.mean(res_test_PLA[:,1], axis=0)
	r_coeff_test_PLA = r2_score(y_test_PLA, pred_PLA_round_test)
	MSE_test_PLA = mean_squared_error(y_test_PLA,pred_PLA_round_test)
	agreement_sum_test_PLA = np.sum(prediction_equal_test_PLA)
	agreement_test_PLA = agreement_sum_test_PLA/len(y_test_PLA)

	#Genotype
	pred_genotype_test = np.empty([res_test_size, 1], dtype = int)
	for i in range(len(predictions_test_genotype)):
		pred_genotype_test[i] = np.argmax(predictions_test_genotype[i])

	test_genotype_test = np.empty([res_test_size, 1], dtype=int)
	for i in range(len(y_test_genotype)):
		test_genotype_test[i] = np.argmax(y_test_genotype[i])
	y_test_genotype = np.copy(test_genotype_test)
	# y_train_genotype = np.reshape(y_train_genotype, (len(y_train_genotype),1))
	# y_test_genotype = np.reshape(y_test_genotype, (len(y_test_genotype),1))
	res_test_genotype = np.concatenate((y_test_genotype, pred_genotype_test) , axis =1)

	prediction_equal_test_genotype = np.equal(res_test_genotype[:,0],res_test_genotype[:,1])
	prediction_equal_test_genotype = prediction_equal_test_genotype.astype(int)
	prediction_equal_test_genotype = np.reshape(prediction_equal_test_genotype, (len(res_test_genotype),1))

	res_test_genotype = np.concatenate((res_test_genotype,prediction_equal_test_genotype), axis=1)

	agreement_sum_test_genotype = np.sum(prediction_equal_test_genotype)
	agreement_test_genotype = agreement_sum_test_genotype/len(y_test_genotype)


	res_count_equal = np.empty([3,res_total_size])
	res_count_equal[0][0:res_test_size] = res_test_count[:,0]
	res_count_equal[1][0:res_test_size] = res_test_count[:,1]
	res_count_equal[2][0:res_test_size] = res_test_count[:,2]

	res_PLA_equal = np.empty([3,res_total_size])
	res_PLA_equal[0][0:res_test_size] = res_test_PLA[:,0]
	res_PLA_equal[1][0:res_test_size] = res_test_PLA[:,1]
	res_PLA_equal[2][0:res_test_size] = res_test_PLA[:,2]

	res_genotype_equal = np.empty([3,res_total_size])
	res_genotype_equal[0][0:res_test_size] = res_test_genotype[:,0]
	res_genotype_equal[1][0:res_test_size] = res_test_genotype[:,1]
	res_genotype_equal[2][0:res_test_size] = res_test_genotype[:,2]


	print('Writing Statistics files')
	# Statististics Count
	results_count_dict = OrderedDict()
	results_count_dict['DIC train'] = average_diff_train_count
	results_count_dict['STD DIC train'] = difference_std_count_train
	results_count_dict['|DIC| train'] = average_diff_abs_train_count
	results_count_dict['STD |DIC| train'] = difference_std_abs_count_train
	results_count_dict['MSE train'] = MSE_train
	results_count_dict['R^2 train'] = r_coeff_train
	results_count_dict['Agreement train'] = agreement_train
	results_count_dict['DIC'] = average_diff_test_count
	results_count_dict['STD DIC'] = difference_std_count_test
	results_count_dict['|DIC|'] = average_diff_abs_test_count
	results_count_dict['STD |DIC|'] = difference_std_abs_count_test
	results_count_dict['MSE'] = MSE_test
	results_count_dict['R^2'] = r_coeff_test
	results_count_dict['Agreement test'] = agreement_test

	# Statististics Age
	results_PLA_dict = OrderedDict()
	results_PLA_dict['DIA train'] = average_diff_train_PLA
	results_PLA_dict['STD DIA train'] = difference_std_PLA_train
	results_PLA_dict['|DIA| train'] = average_diff_abs_train_PLA
	results_PLA_dict['STD |DIA| train'] = difference_std_abs_PLA_train
	results_PLA_dict['MSE train'] = MSE_train_PLA
	results_PLA_dict['R^2 train'] = r_coeff_train_PLA
	results_PLA_dict['Agreement train'] = agreement_train_PLA
	results_PLA_dict['DIA'] = average_diff_test_PLA
	results_PLA_dict['STD DIA'] = difference_std_PLA_test
	results_PLA_dict['|DIA|'] = average_diff_abs_test_PLA
	results_PLA_dict['STD |DIA|'] = difference_std_abs_PLA_test
	results_PLA_dict['MSE'] = MSE_test_PLA
	results_PLA_dict['R^2'] = r_coeff_test_PLA
	results_PLA_dict['Agreement test'] = agreement_test_PLA

	#Statistics Genotype
	results_genotype_dict = OrderedDict()
	results_genotype_dict['Agreement train'] = agreement_train_genotype
	results_genotype_dict['Agreement test'] = agreement_test_genotype


	results_arrays_dict_count = OrderedDict()
	# results_arrays_dict_count['Training Image name'] = result_arr_train[:,2]
	results_arrays_dict_count['Training targets'] = res_train_count[:,0]
	results_arrays_dict_count['Training predictions'] = res_train_count[:,1]
	results_arrays_dict_count['Training agreement'] = res_train_count[:,2]
	# results_arrays_dict_count['Test Image name'] = results_arr_equal[:,2]
	results_arrays_dict_count['Test targets'] = res_count_equal[0,:]
	results_arrays_dict_count['Test predictions'] = res_count_equal[1,:]
	results_arrays_dict_count['Test agreement'] = res_count_equal[2,:]

	results_arrays_dict_PLA = OrderedDict()
	# results_arrays_dict_PLA['Training Image name'] = result_arr_train[:,2]
	results_arrays_dict_PLA['Training targets'] = res_train_PLA[:,0]
	results_arrays_dict_PLA['Training predictions'] = res_train_PLA[:,1]
	results_arrays_dict_PLA['Training agreement'] = res_train_PLA[:,2]
	# results_arrays_dict_PLA['Test Image name'] = results_arr_equal[:,2]
	results_arrays_dict_PLA['Test targets'] = res_PLA_equal[0,:]
	results_arrays_dict_PLA['Test predictions'] = res_PLA_equal[1,:]
	results_arrays_dict_PLA['Test agreement'] = res_PLA_equal[2,:]

	results_arrays_dict_genotype = OrderedDict()
	# results_arrays_dict_genotype['Training Imgenotype name'] = result_arr_train[:,2]
	results_arrays_dict_genotype['Training targets'] = res_train_genotype[:,0]
	results_arrays_dict_genotype['Training predictions'] = res_train_genotype[:,1]
	results_arrays_dict_genotype['Training agreement'] = res_train_genotype[:,2]
	# results_arrays_dict_genotype['Test Imgenotype name'] = results_arr_equal[:,2]
	results_arrays_dict_genotype['Test targets'] = res_genotype_equal[0,:]
	results_arrays_dict_genotype['Test predictions'] = res_genotype_equal[1,:]
	results_arrays_dict_genotype['Test agreement'] = res_genotype_equal[2,:]



	results_arrays_dataframe_count = pd.DataFrame(results_arrays_dict_count, index=list(range(0,len(y_train_count))))
	excel_writer_one = pd.ExcelWriter(results_path+'/ResultsPredictionsCount.xlsx', engine='xlsxwriter')
	results_arrays_dataframe_count.to_excel(excel_writer_one, sheet_name='Sheet1')
	excel_writer_one.save()

	results_arrays_dataframe_PLA = pd.DataFrame(results_arrays_dict_PLA, index=list(range(0,len(y_train_PLA))))
	excel_writer_one = pd.ExcelWriter(results_path+'/ResultsPredictionsAge.xlsx', engine='xlsxwriter')
	results_arrays_dataframe_PLA.to_excel(excel_writer_one, sheet_name='Sheet1')
	excel_writer_one.save()

	results_arrays_dataframe_genotype = pd.DataFrame(results_arrays_dict_genotype, index=list(range(0,len(y_train_genotype))))
	excel_writer_one = pd.ExcelWriter(results_path+'/ResultsPredictionsGenotype.xlsx', engine='xlsxwriter')
	results_arrays_dataframe_genotype.to_excel(excel_writer_one, sheet_name='Sheet1')
	excel_writer_one.save()

	results_dataframe_count = pd.DataFrame(results_count_dict, index=[0])
	excel_writer_two = pd.ExcelWriter(results_path+'/ResultsStatsCount.xlsx', engine='xlsxwriter')
	results_dataframe_count.to_excel(excel_writer_two, sheet_name='Sheet1')
	excel_writer_two.save()

	results_dataframe_PLA = pd.DataFrame(results_PLA_dict, index=[0])
	excel_writer_two = pd.ExcelWriter(results_path+'/ResultsStatsAge.xlsx', engine='xlsxwriter')
	results_dataframe_PLA.to_excel(excel_writer_two, sheet_name='Sheet1')
	excel_writer_two.save()

	results_dataframe_genotype = pd.DataFrame(results_genotype_dict, index=[0])
	excel_writer_two = pd.ExcelWriter(results_path+'/ResultsStatsGenotype.xlsx', engine='xlsxwriter')
	results_dataframe_genotype.to_excel(excel_writer_two, sheet_name='Sheet1')
	excel_writer_two.save()

	# #print('Plants used for training are', train_plants, 'for validation', validate_plants, 'for testing' , test_plants)
	# # print(difference_arr)
	# # print(result_arr)
	# print('Average difference is', average_diff_test_count, 'Test count MSE is ' ,MSE_test, 'Genotype accuracy', agreement_test_genotype)
	# print('Average difference is', average_diff_test_count, 'Test count MSE is ' ,MSE_test)
	#return predictions_train, predictions_test

