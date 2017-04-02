# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 19:13:36 2017

@author: Weidi Xie

@description: 
This is the file to create the model, similar as the paper, 
but with batch normalization,
make it more easier to train.

U-net version is also provided.
"""

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import (
    Input,
    Activation,
    Merge,
    merge,
    Dropout,
    Reshape,
    Permute,
    Dense,
    UpSampling2D,
    Flatten
    )
from keras.optimizers import SGD, RMSprop
from keras.layers.convolutional import (
    Convolution2D)
from keras.layers.pooling import (
    MaxPooling2D,
    AveragePooling2D
    )
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

weight_decay = 1e-5
K.set_image_dim_ordering('tf')

def _conv_bn_relu(nb_filter, row, col, subsample = (1,1)):
    def f(input):
        conv_a = Convolution2D(nb_filter, row, col, subsample = subsample,
                               init = 'orthogonal', 
                               border_mode='same', bias = False)(input)
        norm_a = BatchNormalization()(conv_a)
        act_a = Activation(activation = 'relu')(norm_a)
        return act_a
    return f
    
def _conv_bn_relu_x2(nb_filter, row, col, subsample = (1,1)):
    def f(input):
        conv_a = Convolution2D(nb_filter, row, col, subsample = subsample,
                               init = 'orthogonal', border_mode = 'same',bias = False,
                               W_regularizer = l2(weight_decay),
                               b_regularizer = l2(weight_decay))(input)
        norm_a = BatchNormalization()(conv_a)
        act_a = Activation(activation = 'relu')(norm_a)
        conv_b = Convolution2D(nb_filter, row, col, subsample = subsample,
                               init = 'orthogonal', border_mode = 'same',bias = False,
                               W_regularizer = l2(weight_decay),
                               b_regularizer = l2(weight_decay))(act_a)
        norm_b = BatchNormalization()(conv_b)
        act_b = Activation(activation = 'relu')(norm_b)
        return act_b
    return f

def FCRN_A_base(input):
    block1 = _conv_bn_relu(32,3,3)(input)
    pool1 = MaxPooling2D(pool_size=(2,2))(block1)
    # =========================================================================
    block2 = _conv_bn_relu(64,3,3)(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(block2)
    # =========================================================================
    block3 = _conv_bn_relu(128,3,3)(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(block3)
    # =========================================================================
    block4 = _conv_bn_relu(512,3,3)(pool3)
    # =========================================================================
    up5 = UpSampling2D(size=(2, 2))(block4)
    block5 = _conv_bn_relu(128,3,3)(up5)
    # =========================================================================
    up6 = UpSampling2D(size=(2, 2))(block5)
    block6 = _conv_bn_relu(64,3,3)(up6)
    # =========================================================================
    up7 = UpSampling2D(size=(2, 2))(block6)
    block7 = _conv_bn_relu(32,3,3)(up7)
    return block7

def FCRN_A_base_v2(input):
    block1 = _conv_bn_relu_x2(32,3,3)(input)
    pool1 = MaxPooling2D(pool_size=(2,2))(block1)
    # =========================================================================
    block2 = _conv_bn_relu_x2(64,3,3)(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(block2)
    # =========================================================================
    block3 = _conv_bn_relu_x2(128,3,3)(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(block3)
    # =========================================================================
    block4 = _conv_bn_relu(512,3,3)(pool3)
    # =========================================================================
    up5 = UpSampling2D(size=(2, 2))(block4)
    block5 = _conv_bn_relu_x2(128,3,3)(up5)
    # =========================================================================
    up6 = UpSampling2D(size=(2, 2))(block5)
    block6 = _conv_bn_relu_x2(64,3,3)(up6)
    # =========================================================================
    up7 = UpSampling2D(size=(2, 2))(block6)
    block7 = _conv_bn_relu_x2(32,3,3)(up7)
    return block7

def U_net_base(input, nb_filter = 64):
    block1 = _conv_bn_relu_x2(nb_filter,3,3)(input)
    pool1 = MaxPooling2D(pool_size=(2,2))(block1)
    # =========================================================================
    block2 = _conv_bn_relu_x2(nb_filter,3,3)(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(block2)
    # =========================================================================
    block3 = _conv_bn_relu_x2(nb_filter,3,3)(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(block3)
    # =========================================================================
    block4 = _conv_bn_relu_x2(nb_filter,3,3)(pool3)
    up4 = merge([UpSampling2D(size=(2, 2))(block4), block3], mode='concat', concat_axis=-1)
    # =========================================================================
    block5 = _conv_bn_relu_x2(nb_filter,3,3)(up4)
    up5 = merge([UpSampling2D(size=(2, 2))(block5), block2], mode='concat', concat_axis=-1)
    # =========================================================================
    block6 = _conv_bn_relu_x2(nb_filter,3,3)(up5)
    up6 = merge([UpSampling2D(size=(2, 2))(block6), block1], mode='concat', concat_axis=-1)
    # =========================================================================
    block7 = _conv_bn_relu(nb_filter,3,3)(up6)
    return block7

def buildModel_FCRN_A (input_dim):
    input_ = Input (shape = (input_dim))
    # =========================================================================
    act_ = FCRN_A_base (input_)
    # =========================================================================
    density_pred =  Convolution2D(1, 1, 1, bias = False, activation='linear',\
                                  init='orthogonal',name='pred',border_mode='same')(act_)
    # =========================================================================
    model = Model (input = input_, output = density_pred)
    opt = SGD(lr = 1e-2, momentum = 0.9, nesterov = True)
    model.compile(optimizer = opt, loss = 'mse')
    return model

def buildModel_FCRN_A_v2 (input_dim):
    input_ = Input (shape = (input_dim))
    # =========================================================================
    act_ = FCRN_A_base_v2 (input_)
    # =========================================================================
    density_pred =  Convolution2D(1, 1, 1, bias = False, activation='linear',\
                                  init='orthogonal',name='pred',border_mode='same')(act_)
    # =========================================================================
    model = Model (input = input_, output = density_pred)
    opt = SGD(lr = 1e-2, momentum = 0.9, nesterov = True)
    model.compile(optimizer = opt, loss = 'mse')
    return model

def buildModel_U_net (input_dim):
    input_ = Input (shape = (input_dim))
    # =========================================================================
    act_ = U_net_base (input_, nb_filter = 64 )
    # =========================================================================
    density_pred =  Convolution2D(1, 1, 1, bias = False, activation='linear',\
                                  init='orthogonal',name='pred',border_mode='same')(act_)
    # =========================================================================
    model = Model (input = input_, output = density_pred)
    opt = RMSprop(1e-3)
    model.compile(optimizer = opt, loss = 'mse')
    return model
