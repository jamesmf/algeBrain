# -*- coding: utf-8 -*-
"""
Created on Sun May  7 00:26:14 2017

@author: jmf
"""
from __future__ import print_function
from __future__ import absolute_import
import generateData as gen
import numpy as np
import keras.backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import RMSprop, SGD
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, Dropout
from keras.layers import Activation, ZeroPadding2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D, BatchNormalization
from keras.models import Model
import warnings
from keras.layers import Input
from keras import layers
from keras import backend as K


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
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
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
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
    return x


def ResNet(pooling='avg'):
    """
    # Returns
        A Keras model instance.
    """

    # Determine proper input shape
    input_shape = (128,128,1)

    img_input = Input(shape=input_shape)
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

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if pooling == 'avg':
        x = GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D()(x)
    
    newOutput = Flatten()(x)
    
    # Glyph Recognizer Output
    glyphOutput = Dense(gen.MULTITASK_SIZE, name='glyphs')(newOutput)
    
    newOutput = Dense(256,activation='tanh')(newOutput)
    newOutput = Dropout(0.5)(newOutput)
    newOutput = Dense(256,activation='tanh')(newOutput)    
    answerLayer = Dense(1, name='Prediction')(newOutput)
    
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    inputs = img_input
    
    # Create model.
    model = Model(inputs=inputs,outputs=[answerLayer,glyphOutput])
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                                loss='mean_squared_error',
                                loss_weights={'glyphs':10,
                                              'Prediction':1})

    return model
    
    
#def newModel():
#    #use the first 6 layers of Inception V3
#    inp = Input(shape=(gen.MAXHEIGHT,gen.MAXWIDTH,3))
#    baseModel = InceptionV3(input_tensor=inp, weights='imagenet', include_top=False)
#    #make inception layers untrainable to begin
#    for layer in baseModel.layers:
#        layer.trainable = False
#    originalOut = baseModel.get_layer('mixed5').output
#    newOutput = Flatten()(originalOut)
#    newOutput = Dense(256,activation='tanh')(newOutput)
#    newOutput = Dropout(0.5)(newOutput)
#    newOutput = Dense(256,activation='tanh')(newOutput)
#    
#    #to start, we'll do naive 'autoencoding' (no convolutions, just dense)
#    newOutput = Dense(gen.MAXHEIGHT*gen.MAXWIDTH)(newOutput)
#    model = Model(inputs=baseModel.input,output=newOutput)
#    model.compile(optimizer='rmsprop', loss='mean_squared_error')
#    return model
#
#def recreateModel(model):
#    for layer in model.layers:
#        layer.trainable = True
#    lastHiddenLayer = model.layers[-1].output
#    lastVisualLayer = model.layers[-5].output
#    answerLayer = Dense(1)(lastHiddenLayer)
#    multitaskLayer = Dense(gen.MULTITASK_SIZE)(lastVisualLayer)
#    model = Model(inputs=model.input,outputs=[answerLayer,multitaskLayer])
#    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
#                              loss='mean_squared_error',
#                              loss_weights=[1,5])
#    return model
