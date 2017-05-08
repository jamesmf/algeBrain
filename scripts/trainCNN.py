# -*- coding: utf-8 -*-
"""
Created on Sun May  7 12:51:59 2017

@author: jmf
"""

import generateData as gen
import defineModel as dm
import numpy as np
import scipy.misc as mi


#model = dm.newModel()
X,y = gen.getFullMatrix("simple")
#print(X.shape)
#print(X[0])
mi.imshow(X[0])
#model.fit(X,X[:,:,:,0].reshape(X.shape[0],X.shape[1]*X.shape[2]))