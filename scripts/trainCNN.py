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
from keras.callbacks import EarlyStopping, ModelCheckpoint
import sys

patience = 10
numEpochs = 250
modelPath = "../models/"

callbacks = [
    EarlyStopping(monitor='val_Prediction_loss', 
                  patience=patience, verbose=1),
    ModelCheckpoint(modelPath+'algeBrain.cnn', 
                    monitor='val_loss', save_best_only=True,
                    verbose=1)
]

model = dm.ResNet50(include_top=False, weights='imagenet',
             input_tensor=None, input_shape=None,
             pooling=None,
             classes=1000)

Xtrain, ytrain = gen.getFullMatrix("simple3", trivialSupplement=500)
Xcv, ycv = gen.getFullMatrix("simpleValidation",trivialSupplement=100)
cbs = model.fit(Xtrain,ytrain,epochs=numEpochs,
                validation_data=(Xcv, ycv),
                callbacks=callbacks,verbose=1,shuffle=True)
model = load_model(modelPath+'algeBrain.cnn')
    

print("loading model")
model = load_model(modelPath+'algeBrain.cnn')
print("model loaded")
Xtest,ytest = gen.getFullMatrix("simpleTest",trivialSupplement=100)
ypred = model.predict(Xtest)
yMultiTest = ytest[1]
ytest = ytest[0]
for i in range(0,Xtest.shape[0]):
    print(ypred[0][i],ytest[i],'\t\t',ypred[1][i],yMultiTest[i])
    mi.imsave("../data/output/im_"+str(i)+"_"+str(ytest[i])+'_'+str(ypred[0][i])+'.jpeg',Xtest[i,:,:,0])
df = pd.DataFrame(data=ytest,columns=["ytest"])
df["ypred"] = ypred[0]
df = df.reset_index()
df["SE"] = (df["ytest"] - df["ypred"])**2
