# -*- coding: utf-8 -*-
"""
Created on Sun May  7 12:51:59 2017

@author: jmf
"""

import generateData as gen
import defineModel as dm
import numpy as np
from sklearn.utils import shuffle
import scipy.misc as mi
from keras import optimizers
from keras.layers import Dense
from keras.models import Model
from keras.models import load_model
import sys

numEpochs = 10


if len(sys.argv) < 2:
    model = dm.newModel()
    X,y = gen.getFullMatrix("simple")
    X = X[:6000]
    y = y[:6000]
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
        print("generating new data")
        X,y = gen.getFullMatrix("simple")
        X,y = shuffle(X,y,random_state=0)
        model.fit(X,y)
        model.save('../models/simpleModel.h5')
    
elif sys.argv[1] == "train":
    print("loading model")
    model = load_model('../models/simpleModel.h5')
    print("model loaded")
    for epoch in range(0,numEpochs):
        print("generating new data")
        X,y = gen.getFullMatrix("simple",trivialSupplement=15000)
        X,y = shuffle(X,y,random_state=0)
        model.fit(X,y)
        model.save('../models/simpleModel.h5')
    
elif sys.argv[1] == "test":
    print("loading model")
    model = load_model('../models/simpleModel.h5')
    print("model loaded")
    Xtest,ytest = gen.getFullMatrix("simpleTest",trivialSupplement=100)
    ypred = model.predict(Xtest)
    for i in range(0,Xtest.shape[0]):
        mi.imsave("../data/output/im_"+str(i)+"_"+str(ytest[i][0])+'_'+str(ypred[i])+'.jpeg',Xtest[i,:,:,:])
    