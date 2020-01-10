import os
import traceback
import scipy.misc as misc
import skimage.transform as skt
import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
import random
from keras.utils import to_categorical
from PIL import Image, ImageOps


def load_data_mixed(data_path):

	path_img = data_path+'/IndividualPlantsCrop/'
	path_img_20_min = '/home/andrei/tensorflow/Plantsexp/valerio_dataset/20mcomplete/IndividualPlants/'
	path_labels = '/home/andrei/tensorflow/Plantsexp/valerio_dataset/'

	path_img_name = np.array([glob.glob(path_img + 'Plant_' + str(h + 1) + '/*.png') for h in range(24)])
	path_img_name = np.array([np.sort(path_img_name)[h][0::2] for h in range(24)])

	path_img_name_20_min = np.array([glob.glob(path_img_20_min + 'Plant_' + str(h + 1) + '/*.png') for h in range(24)])
	path_img_name_20_min = np.array([np.sort(path_img_name_20_min)[h] for h in range(24)])
	path_img_name_20_min = np.array([path_img_name_20_min[h][108:1044] for h in range(24)])
	path_img_name_20_min = np.array([path_img_name_20_min[h][0::6] for h in range(24)])

	path_mask_name = np.array([glob.glob(path_img + 'Plant_' + str(h + 1) + '/*fg.png') for h in range(24)])
	path_mask_name = np.sort(path_mask_name)

	gen_col = [5, 7, 18, 20, 23]
	gen_not_col = [1, 3, 8, 11, 15, 2, 6, 13, 21, 24, 4, 9, 12, 16, 19, 10, 14, 17, 22]


	# get_images = np.empty([24,52,317,309,3])
	get_images_col = np.empty(5, dtype=object)
	get_images_count = np.empty(24, dtype=object)

	get_img_names = np.empty(24, dtype=object)
	get_masks = np.empty(24, dtype=object)

	for i in range(5):
		get_images_col[i] = np.array([np.array(Image.open(fname)) for fname in path_img_name_20_min[gen_col[i] - 1]])

	for i in range(24):
		get_images_count[i] = np.array([np.array(Image.open(fname)) for fname in path_img_name[i]])
		get_masks[i] = np.array([np.array(Image.open(fname)) for fname in path_mask_name[i]])

	print('Done importing images')

	genotype_ein2 = [1, 3, 8, 11, 15]
	genotype_pgm = [2, 6, 13, 21, 24]
	genotype_ctr = [4, 9, 12, 16, 19]
	genotype_col = [5, 7, 18, 20, 23]
	genotype_adh = [10, 14, 17, 22]

	rand_ein2 = random.choice(genotype_ein2)
	idx_ein2 = gen_not_col.index(rand_ein2)

	rand_pgm = random.choice(genotype_pgm)
	idx_pgm = gen_not_col.index(rand_pgm)

	rand_ctr = random.choice(genotype_ctr)
	idx_ctr = gen_not_col.index(rand_ctr)

	rand_col = random.choice(genotype_col)
	idx_col = genotype_col.index(rand_col)

	rand_adh = random.choice(genotype_adh)
	idx_adh = gen_not_col.index(rand_adh)

	train_ein2 = genotype_ein2.remove(rand_ein2)
	train_pgm = genotype_pgm.remove(rand_pgm)
	train_ctr = genotype_ctr.remove(rand_ctr)
	train_col = genotype_col.remove(rand_col)
	train_adh = genotype_adh.remove(rand_adh)

	train_set = []
	train_set.extend(genotype_ein2)
	train_set.extend(genotype_pgm)
	train_set.extend(genotype_ctr)
	train_set.extend(genotype_col)
	train_set.extend(genotype_adh)
	train_set.sort()
	# Create a training set
	train_set = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16, 18, 19, 21, 22, 23] 
	train_set = [x - 1 for x in train_set]

	train_set_no_col = []
	train_set_no_col.extend(genotype_ein2)
	train_set_no_col.extend(genotype_pgm)
	train_set_no_col.extend(genotype_ctr)
	train_set_no_col.extend(genotype_adh)
	train_set_no_col.sort()
	train_set_no_col = [x - 1 for x in train_set_no_col]

	test_set_count = [rand_ein2, rand_pgm, rand_ctr, rand_col, rand_adh]
	test_set_count.sort()
	# Create a test set
	test_set_count = [9, 11, 17, 20, 24]
	test_set_count = [x - 1 for x in test_set_count]

	test_set_gen = [rand_ein2, rand_pgm, rand_ctr, rand_adh, rand_col]
	test_set_gen.extend(genotype_col)
	test_set_gen.sort()
	test_set_gen = [x - 1 for x in test_set_gen]

	ppp = [x + 1 for x in test_set_count]
	print('Test set is ', ppp)

	test_idx_count = [idx_ein2, idx_pgm, idx_ctr, idx_col, idx_adh]

	test_imgs_count_i = get_images_count[[test_set_count]]
	test_masks_count = get_masks[[test_set_count]]

	test_imgs_gen = get_images_count[[test_set_count]]
	test_masks_gen = get_masks[[test_set_count]]

	train_imgs_count_i = np.delete(get_images_count, test_set_count, 0)
	train_masks_count = np.delete(get_masks, test_set_count, 0)

	train_imgs_gen_i = np.delete(get_images_count, test_set_gen, 0)

	train_imgs_col_i = np.delete(get_images_col, idx_col, 0)
	train_imgs_col = np.concatenate(train_imgs_col_i, axis=0)


	train_imgs_count = np.concatenate(train_imgs_count_i, axis=0)
	train_masks_count = np.concatenate(train_masks_count, axis=0)

	train_imgs_gen = np.concatenate(train_imgs_gen_i, axis=0)

	train_imgs_col = np.array([misc.imresize(train_imgs_col[i], [317, 309, 3]) for i in range(0, len(train_imgs_col))])

	train_imgs_gen_all = np.concatenate([train_imgs_gen, train_imgs_col], axis=0)

	test_imgs_count = np.concatenate(test_imgs_count_i, axis=0)

	# Load and process labels

	# Load PLA labels
	mask_training = np.empty(19, dtype=object)
	# mask_val = np.empty(5, dtype = object)
	mask_training_gen = np.empty(15, dtype=object)
	mask_testing = np.empty(5, dtype=object)

	PLA_training = np.empty([19, 52], dtype=int)
	PLA_training_gen = np.empty([15, 52], dtype=int)
	PLA_training_gen_col = np.empty([4, 52], dtype=int)
	PLA_training_gen_col_two = np.empty([4, 156], dtype=int)
	PLA_testing = np.empty([5, 52], dtype=int)

	step = 0
	for i in train_set:
		mask_training[step] = get_masks[i]
		for j in range(len(mask_training[step])):
			PLA_training[step][j] = np.sum(mask_training[step][j])
		step += 1

	step = 0
	for i in train_set_no_col:
		mask_training_gen[step] = get_masks[i]
		for j in range(len(mask_training_gen[step])):
			PLA_training_gen[step][j] = np.sum(mask_training_gen[step][j])
		step += 1

	step = 0
	for i in genotype_col:
		for j in range(len(mask_training_gen[step])):
			PLA_training_gen_col[step][j] = np.sum(mask_training_gen[step][j])
		step += 1

	for i in range(4):
		for j in range(0,156,3):
			PLA_training_gen_col_two[i][j] = PLA_training_gen_col[i][int(j / 3)]
			PLA_training_gen_col_two[i][j+1] = PLA_training_gen_col[i][int(j / 3)]
			PLA_training_gen_col_two[i][j+2] = PLA_training_gen_col[i][int(j / 3)]


	step = 0
	for i in test_set_count:
		mask_testing[step] = get_masks[i]
		for j in range(len(mask_testing[step])):
			PLA_testing[step][j] = np.sum(mask_testing[step][j])
		step += 1

	PLA_training = np.concatenate(PLA_training, axis=0)
	PLA_training_gen = np.concatenate(PLA_training_gen, axis=0)
	PLA_training_gen_col_two = np.concatenate(PLA_training_gen_col_two, axis=0)
	PLA_training_gen = np.concatenate([PLA_training_gen, PLA_training_gen_col_two], axis =0)
	PLA_testing = np.concatenate(PLA_testing, axis=0)

	PLA_labels = np.empty(3, dtype=object)
	PLA_labels[0] = PLA_training
	PLA_labels[1] = PLA_testing
	PLA_labels[2] = PLA_training_gen

	# Load count labels
	count_all = pd.read_csv(path_labels + 'plant_annotations.csv', header=None)
	count_all = np.transpose(np.array(count_all))

	# Training
	training_count = np.empty(0)
	for i in train_set:
		training_count = np.concatenate([training_count, count_all[i]])
	print(training_count.shape)

	training_count_gen = np.empty(0)
	for i in train_set_no_col:
		training_count_gen = np.concatenate([training_count_gen, count_all[i]])

	training_count_gen_col = np.empty([4, 52])
	training_count_gen_col_two = np.empty([4, 156])
	step = 0
	for i in genotype_col:
		for j in range(52):
			training_count_gen_col[step][j] = count_all[i][j]
		step += 1

	for j in range(4):
		for i in range(0, 156, 3):
			training_count_gen_col_two[j][i] = training_count_gen_col[j][int(i/3)]
			training_count_gen_col_two[j][i + 1] = training_count_gen_col[j][int(i/3)]
			training_count_gen_col_two[j][i + 2] = training_count_gen_col[j][int(i/3)]

	training_count_gen_col = np.concatenate(training_count_gen_col, axis=0)
	training_count_gen_col_two = np.concatenate(training_count_gen_col_two, axis=0)
	training_count_gen = np.concatenate([training_count_gen, training_count_gen_col_two], axis=0)

	# testing
	testing_count = np.empty(0)
	for i in test_set_count:
		testing_count = np.concatenate([testing_count, count_all[i]])

	count_labels = np.empty(3, dtype=object)
	count_labels[0] = training_count
	count_labels[1] = testing_count
	count_labels[2] = training_count_gen
	# count_labels = np.vstack((training_count, val_count, testing_count)).T

	# Load Genotype labels
	genotype_all = pd.read_csv(path_labels + 'plant_genotypes.csv', header=None)
	genotype_all = np.transpose(np.array(genotype_all))

	# training
	training_genotype = []
	for i in train_set:
		for h in range(52):
			training_genotype.append(genotype_all[2, i])
	training_genotype = np.array(training_genotype)
	training_genotype[training_genotype > 1] = 0
	training_genotype_hot = to_categorical(training_genotype)
	# training_genotype_hot = training_genotype_hot[:,1:]

	training_genotype_col = np.ones([624, 1])
	training_genotype_not_col = np.zeros([780, 1])
	training_genotype_all = np.concatenate([training_genotype_not_col, training_genotype_col], axis=0)
	training_genotype_all_hot = to_categorical(training_genotype_all)

	# Testing
	testing_genotype = []
	for i in test_set_count:
		for h in range(52):
			testing_genotype.append(genotype_all[2, i])
	testing_genotype = np.array(testing_genotype)
	testing_genotype[testing_genotype > 1] = 0
	testing_genotype_hot = to_categorical(testing_genotype)
	# testing_genotype_hot = testing_genotype_hot[:,1:]

	# testing_genotype_col = np.ones([260,1])
	# testing_genotype_not_col = np.zeros([260,1])
	# testing_genotype_all = np.concatenate([testing_genotype_not_col, testing_genotype_col], axis=0)
	# testing_genotype_all_hot = to_categorical(testing_genotype_all)

	genotype_labels = np.empty(3, dtype=object)
	genotype_labels[0] = training_genotype
	genotype_labels[1] = testing_genotype
	genotype_labels[2] = np.squeeze(training_genotype_all)

	sets = []
	train_set = [x + 1 for x in train_set]
	test_set_count = [x + 1 for x in test_set_count]
	train_set_no_col = [x + 1 for x in train_set_no_col]
	sets.append(train_set)
	sets.append(test_set_count)
	sets.append(train_set_no_col)


	imgs_sets_i = []
	imgs_sets_i.append(train_imgs_count_i)
	imgs_sets_i.append(test_imgs_count_i)
	imgs_sets_i.append(train_imgs_gen_i)
	imgs_sets_i.append(get_images_count)
	imgs_sets_i.append(get_images_col)



	print('Done processing labels')
	# print(train_imgs_count.shape)
	# print(train_imgs_gen_all.shape)
	# print(test_imgs_count.shape)
	# print(count_labels[0].shape)
	# print(genotype_labels[2].shape)
	# print(PLA_labels[1].shape)

	return train_imgs_count, train_imgs_gen_all, test_imgs_count, count_labels, genotype_labels, PLA_labels, sets, imgs_sets_i


