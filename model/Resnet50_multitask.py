from __future__ import print_function
from __future__ import absolute_import

import warnings
import keras
import keras.backend as K
from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras import regularizers
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils import to_categorical
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
# from keras.applications.imagenet_utils import _obtain_input_shape

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
# import cv2
import matplotlib.pyplot as plt
import random



WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    # A block that has a conv layer at the skip connection shortcut.
    
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    # Returns Output tensor for the block.
    return x


def ResNet50(include_top=False, weights='imagenet',
             input_tensor=None, input_shape=None,
             pooling=None):
    #Instantiates the ResNet50 architecture.
   
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    # # Determine proper input shape
    # input_shape = _obtain_input_shape(input_shape,
    #                                   default_size=224,
    #                                   min_size=197,
    #                                   data_format=K.image_data_format(),
    #                                   include_top=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if pooling == 'avg':
        x = GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D()(x)


    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet50')

    # load weights
    if weights == 'imagenet':
        weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models',
                                md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)
            if include_top:
                maxpool = model.get_layer(name='avg_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1000')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

        if K.image_data_format() == 'channels_first' and K.backend() == 'tensorflow':
            warnings.warn('You are using the TensorFlow backend, yet you '
                          'are using the Theano '
                          'image data format convention '
                          '(`image_data_format="channels_first"`). '
                          'For best performance, set '
                          '`image_data_format="channels_last"` in '
                          'your Keras config '
                          'at ~/.keras/keras.json.')
    return model


# Counter model with NO data augmentation.
def counter_model(x_train_all, x_val_all, y_train_genotype, y_val_genotype, results_path):
    

    res_model = ResNet50(weights='imagenet', include_top=False, input_shape=(321,321,3))
    model_res = res_model.output
    model_flat = Flatten(name='flatten')(model_res)
    model = Dense(1024, activation='relu', activity_regularizer= regularizers.l2(0.10))(model_flat)
    model_genotype = Dense(5, activation='softmax')(model)



    input = Input(shape=(128, 128, 3), name='input')
    model = Conv2D(32, 5, strides=(1, 1), padding="same", activation="relu")(input)
    model = MaxPooling2D(pool_size=(3,3), strides=(2,2))(model)
    model = Conv2D(64, 5, strides=(1, 1), padding="same", activation="relu")(model)
    model = MaxPooling2D(pool_size=(3,3), strides=(2,2))(model)
    model = Conv2D(64, 5, strides=(1, 1), padding="same", activation="relu")(model)
    model = MaxPooling2D(pool_size=(3,3), strides=(2,2))(model)
    model = Conv2D(64, 5, strides=(1, 1), padding="same", activation="relu")(model)
    model = MaxPooling2D(pool_size=(3,3), strides=(2,2))(model)
    model = Dense(4096, activation='relu')(model)
    model = Dropout(0.5)(model)
    model = Dense(4096, activation='relu')(model)
    model = Dropout(0.5)(model)
    model_genotype = Dense(5, activation='softmax')(model)

    epoch = 50
    csv_logger = keras.callbacks.CSVLogger('training.log', separator=',')
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.03, mode='min', patience=8)

    model = Model(inputs = res_model.input, outputs = model_genotype)
    model.compile(optimizer=Adam(lr=0.0001), loss= 'categorical_crossentropy', metrics = [metrics.categorical_accuracy])
    fitted_model = model.fit(x_train_all, y_train_genotype, epochs=epoch, validation_data=(x_val_all, y_val_genotype), batch_size=16, callbacks= [csv_logger])
    

    return model

