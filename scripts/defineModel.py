# -*- coding: utf-8 -*-
"""
Created on Sun May  7 00:26:14 2017

@author: jmf
"""
from __future__ import print_function
import generateData as gen
import numpy as np
import keras.backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import RMSprop
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, Dropout
from keras.models import Model

#X, y = gen.getFullMatrix("simple")

def newModel():
    #use the first 6 layers of Inception V3
    inp = Input(shape=(gen.MAXHEIGHT,gen.MAXWIDTH,3))
    baseModel = InceptionV3(input_tensor=inp, weights='imagenet', include_top=False)
    #make inception layers untrainable to begin
    for layer in baseModel.layers:
        layer.trainable = False
    originalOut = baseModel.get_layer('mixed5').output
    newOutput = Dense(128)(originalOut)
    newOutput = Dropout(0.5)(newOutput)
    newOutput = Dense(64)(newOutput)
    newOutput = Flatten()(newOutput)
    #to start, we'll do naive 'autoencoding' (no convolutions, just dense)
    newOutput = Dense(gen.MAXHEIGHT*gen.MAXWIDTH)(newOutput)
    model = Model(inputs=baseModel.input,output=newOutput)
    model.compile(optimizer='rmsprop', loss='mean_squared_error')
    return model
