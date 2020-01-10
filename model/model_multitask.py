import keras
from keras.activations import sigmoid, tanh
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, merge, ConvLSTM2D, Reshape
from keras.layers import Input, Convolution2D, MaxPooling2D, LeakyReLU, LSTM, TimeDistributed, Conv2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import metrics
from keras import backend as K
from keras import regularizers
from keras import callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint
import h5py
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import random

from layer_vis import vis_layer
from model.Resnet50_multitask import ResNet50



def multi_task_model_mixed(results_path, data, missing_labels, gen_weight):
    
    x_train = data[0]
    x_test = data[1]
    y_train_count = data[2]
    y_test_count = data[3]
    y_train_genotype = data[4]
    y_test_genotype = data[5]
    y_train_pla = data[6]
    y_test_pla = data[7]

    class_weights = {0: 1.,
                     1: gen_weight}


    x_aug = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,                                                                                                                                                                                                                                             
        )


    mask_value = -1
    missing_type = 'equal'

    if missing_type == 'random':
        # Random missing labels
        TYC = len(y_train_count)
        how_much_mask = missing_labels
        idx_mask = np.random.randint(TYC, size = int(TYC*how_much_mask)) 
        y_train_count[idx_mask] = mask_value
        where_miss = np.where(y_train_count == mask_value)
        np.savetxt(results_path+'/missing_labels.csv', where_miss[0], delimiter=',')
        np.savetxt(results_path+'/train_labels.csv', y_train_count, delimiter=',')
        print('Missing Labels  ', where_miss[0])
    elif missing_type == 'equal':
        # Equal distance missing labels
        TYC = len(y_train_count)
        how_much_mask = missing_labels
        array_idx = np.array(range(TYC))
        array_idx_split = np.array_split(array_idx, int(TYC*0.25))
        idx_mask = []
        idx_coeff = int(4*how_much_mask)
        for i in array_idx_split:
            idx_mask.append(i[0:idx_coeff])
        idx_mask_array = np.array(idx_mask)
        idx_mask_array = idx_mask_array.ravel()
        y_train_count[idx_mask_array] = mask_value
        where_miss = np.where(y_train_count == mask_value)
        np.savetxt(results_path+'/missing_labels.csv', where_miss[0], delimiter=',')
        np.savetxt(results_path+'/train_labels.csv', y_train_count, delimiter=',')
        print('Missing Labels  ', idx_coeff)
    elif missing_type == 'large':
        # Large Plants Missing labels
        TYC = len(y_train_count)
        how_much_mask = missing_labels
        array_idx = np.array(np.arange(TYC))
        array_idx_split = np.array_split(array_idx, 19)
        idx_mask = []
        idx_coeff = int(52*how_much_mask)
        for i in array_idx_split:
        	idx_mask.append(i[:idx_coeff])
        idx_mask_array = np.array(idx_mask)
        idx_mask_array = idx_mask_array.ravel()
        y_train_count[idx_mask_array] = mask_value
        where_miss = np.where(y_train_count == mask_value)
        np.savetxt(results_path+'/missing_labels.csv', where_miss[0], delimiter=',')
        np.savetxt(results_path+'/train_labels.csv', y_train_count, delimiter=',')
        print('Missing Labels  ', idx_coeff)
    else:
        print('No Missing Labels')



    def generate_data_generator(x_aug, x_train_all, y_train_count, y_train_PLA, y_train_genotype_hot):
        batch_size = 6
        seed = random.randint(1, 1000)
        #genX = x_aug.flow(x_train_all, y_train_count, batch_size=6, seed=seed)
        genY1 = x_aug.flow(x_train_all, y_train_count, batch_size=6, seed=seed)
        genY2 = x_aug.flow(x_train_all, y_train_PLA, batch_size=6, seed=seed)
        genY3 = x_aug.flow(x_train_all, y_train_genotype_hot, batch_size=6, seed=seed)
        
        alpha = 0.2
        l = np.random.beta(alpha, alpha, batch_size)
        
        while True:
            Xi, Yi1 = genY1.next()
            Xi, Yi2 = genY2.next()
            Xi, Yi3 = genY3.next()
            yield Xi , [Yi1, Yi2, Yi3]



    def PLA_loss(y_true, y_pred):
        return K.mean(K.square((y_pred - y_true)*10), axis=-1)

    # function to mask the target value so that the model does not see the label value if it is -1
    def MSE_masked_loss(y_true, y_pred):
        mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
        return K.mean(K.square(y_pred*mask - y_true*mask), axis=-1)

    def variable_MSE(alpha):
        def loss(y_true, y_pred):
            loss = K.mean(K.square(y_pred - y_true), axis=-1)
            return loss * alpha
        return loss

    def variable_binary_entropy(beta):
        def loss(y_true, y_pred):
            loss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
            return loss * beta
        return loss

    def mean_pred(y_true, y_pred):
        return K.mean(y_pred)

    def mean_absolute_accuracy(y_true, y_pred):
        return K.mean(K.abs(K.round(y_pred) - y_true), axis=-1)

    def mse_discrete_accuracy(y_true, y_pred):
        return K.mean(K.square(K.round(y_pred) - y_true), axis=-1)

    # Resnet
    res_model = ResNet50(weights='imagenet', include_top=False, input_shape=(317,309,3))
    model_res = res_model.output
    model_flat = Flatten(name='flatten')(model_res)
    # Shared layer 
    model_shared = Dense(1536, activation='relu', name='shared_dense')(model_flat)

    # Leaf count branch
    model_count = Dense(512, activation='relu', activity_regularizer=regularizers.l2(0.04), name='count_dense')(model_shared)
    leaf_pred = Dense(1, name='count')(model_count)

    # PLA branch
    model_PLA = Dense(512, activity_regularizer=regularizers.l2(0.02), name='PLA_dense')(model_shared)
    LR = LeakyReLU(alpha=0.1, name='LR1')(model_PLA)
    PLA_pred = Dense(1, activation='sigmoid', name='PLA_pred')(LR)

    # Genotype branch
    model_genotype = Dense(512, activation='relu',  name='genotype_dense_1')(model_shared)
    model_genotype = Dense(256, activation='relu', activity_regularizer=regularizers.l2(0.02), name='genotype_dense_2')(model_genotype)
    model_genotype = Dense(1, name='gen_pred', activation='sigmoid')(model_genotype)
    
    

    # Callbacks
    csv_logger = keras.callbacks.CSVLogger(results_path+'/training.log', separator=',')
    csv_logger_two = keras.callbacks.CSVLogger(results_path+'/training_two.log', separator=',')
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.03, mode='min', patience=12)
    checkpoint = ModelCheckpoint(results_path+'/checkpoint.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    layers = [5, 62, 86, 149]
    custom_calback = vis_layer(x_train, y_train_count, layers, results_path)

    model = Model(inputs=res_model.input, outputs=[leaf_pred, PLA_pred, model_genotype])


    # Compile Model
    model.compile(optimizer=Adam(lr=0.0001), loss=[MSE_masked_loss, 'mse', 'binary_crossentropy'],
                  metrics={'gen_pred': 'binary_accuracy', 'count': mse_discrete_accuracy})

    print(model.summary())
    
    epoch = 100
    steps = int(len(x_train)/3)

    # Train Model
    model.fit_generator(generate_data_generator(x_aug, x_train, y_train_count, y_train_pla, y_train_genotype), steps_per_epoch=steps, class_weight=[0, 0, class_weights],
                                                epochs=epoch, validation_data=(x_test, [y_test_count, y_test_pla, y_test_genotype]),
                                                callbacks= [csv_logger, early_stop, checkpoint])



    #model.save(results_path+'/the_model.h5')

    return model

def counter_model_augmentation(results_path, data, missing_labels):
    
    x_train = data[0]
    x_test = data[1]
    y_train_count = data[2]
    y_test_count = data[3]

    mask_value = -1

    TYC = len(y_train_count)
    how_much_mask = missing_labels
    idx_mask = np.random.randint(TYC, size = int(TYC*how_much_mask)) 
    y_train_count[idx_mask] = mask_value
    # y_train_count[:int(TYC*0.2)] = mask_value
    where_miss = np.where(y_train_count == mask_value)
    np.savetxt(results_path+'/missing_labels.csv', where_miss[0], delimiter=',')
    np.savetxt(results_path+'/train_labels.csv', y_train_count, delimiter=',')
    print('Missing Labels  ', where_miss[0])

    def MSE_masked_loss(y_true, y_pred):
        mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
        return K.mean(K.square(y_pred*mask - y_true*mask), axis=-1)

    def mse_discrete_accuracy(y_true, y_pred):
        return K.mean(K.square(K.round(y_pred) - y_true), axis=-1)

    x_aug = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        )

    x_aug.fit(x_train)



    res_model = ResNet50(weights='imagenet', include_top=False, input_shape=(317,309, 3))
    model = res_model.output
    model = Flatten(name='flatten')(model)
    model = Dense(1536, activation='relu', name='count_dense1')(model)
    model = Dense(512, activation='relu', activity_regularizer=regularizers.l2(0.04), name='count_dense2')(model)
    leaf_pred = Dense(1, name='count')(model)

    epoch = 100
    steps = int(len(x_train)/3)
    csv_logger = keras.callbacks.CSVLogger(results_path+'/training.log', separator=',')
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.05, mode='min', patience=12)
    checkpoint = ModelCheckpoint(results_path+'/checkpoint.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    model = Model(inputs = res_model.input, outputs = leaf_pred)
    model.compile(optimizer=Adam(lr=0.0001), loss= MSE_masked_loss, metrics={'count': mse_discrete_accuracy})

    fitted_model= model.fit_generator(x_aug.flow(x_train, y_train_count, batch_size=6), steps_per_epoch=steps,
                                                 epochs=epoch, validation_data=(x_test, y_test_count), callbacks= [csv_logger, checkpoint, early_stop])

    model.save(results_path+'/the_model.h5')

    return model


