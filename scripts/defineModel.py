# -*- coding: utf-8 -*-
"""
Created on Sun May  7 00:26:14 2017

@author: jmf
"""
from __future__ import print_function
import generateData as gen
import numpy as np
import keras.backend as K
from keras.optimizers import RMSprop
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten
from keras.models import Model

data = gen.getFullMatrix("simple")
