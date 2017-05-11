# -*- coding: utf-8 -*-
"""
Created on Sun May  7 12:51:59 2017

@author: jmf
"""

import generateData as gen
import defineModel as dm
import numpy as np
import scipy.misc as mi
from keras import optimizers
from keras.layers import Dense
from keras.models import Model
from keras.models import load_model

numEpochs = 8

model = dm.newModel()
X,y = gen.getFullMatrix("simple")
#print(X.shape)
#print(X[0])
#mi.imshow(X[0])
model.fit(X,X[:,:,:,0].reshape(X.shape[0],X.shape[1]*X.shape[2]),epochs=1)

for layer in model.layers:
    layer.trainable = True
    print(layer.name)
lastHiddenLayer = model.layers[-1].output
model2 = Dense(1)(lastHiddenLayer)
model = Model(inputs=model.input,outputs=model2)

model.compile(optimizer=optimizers.SGD(lr=0.0001, momentum=0.9), loss='mean_squared_error')
for epoch in range(0,numEpochs):
    X,y = gen.getFullMatrix("simple")
    model.fit(X,y)
    model.save('../models/simpleModel.h5')