# -*- coding: utf-8 -*-
"""
Created on Sun May  7 12:51:59 2017

@author: jmf
"""

import generateData as gen
import defineModel as dm
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import scipy.misc as mi
from keras import optimizers
from keras.layers import Dense
from keras.models import Model
from keras.models import load_model
import sys

numEpochs = 25


if len(sys.argv) < 2:
    model = dm.newModel()
    X,y = gen.getFullMatrix("simple")
    X = X[:2000]
    y = y[:2000]
    #print(X.shape)
    #print(X[0])
    #mi.imshow(X[0])
    model.fit(X,X[:,:,:,0].reshape(X.shape[0],X.shape[1]*X.shape[2]),epochs=1)
    
    model = dm.recreateModel(model)

    for epoch in range(0,numEpochs):
        print("generating new data")
        X,y = gen.getFullMatrix("simple")
        X,y0,y1 = shuffle(X,y[0],y[1],random_state=0)
        y = [y0,y1]
        model.fit(X,y)
        model.save('../models/simpleModelMultitask.h5')
    
elif sys.argv[1] == "train":
    print("loading model")
    model = load_model('../models/simpleModelMultitask.h5')
    print("model loaded")
    for epoch in range(0,numEpochs):
        print("generating new data")
        X,y = gen.getFullMatrix("simple2",trivialSupplement=10000)
        X,y0,y1 = shuffle(X,y[0],y[1],random_state=0)
        y = [y0,y1]
        model.fit(X,y)
        model.save('../models/simpleModelMultitask.h5')
    
elif sys.argv[1] == "test":
    print("loading model")
    model = load_model('../models/simpleModelMultitask.h5')
    print("model loaded")
    Xtest,ytest = gen.getFullMatrix("simpleTest",trivialSupplement=100)
    ypred = model.predict(Xtest)
    yMultiTest = ytest[1]
    ytest = ytest[0]
    for i in range(0,Xtest.shape[0]):
        print(ypred[0][i],ytest[i],'\t\t',ypred[1][i],yMultiTest[i])
        mi.imsave("../data/output/im_"+str(i)+"_"+str(ytest[i])+'_'+str(ypred[0][i])+'.jpeg',Xtest[i,:,:,:])
    df = pd.DataFrame(data=ytest,columns=["ytest"])
    df["ypred"] = ypred[0]
    df = df.reset_index()
    df["SE"] = (df["ytest"] - df["ypred"])**2
