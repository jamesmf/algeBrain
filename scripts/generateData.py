# -*- coding: utf-8 -*-
"""
Created on Sat May  6 17:31:02 2017

@author: jmf
"""
from __future__ import print_function
import random
import numpy as np
import cairo
import os
import re

MAXWIDTH = 128
MAXHEIGHT = 64
SIZEVARIANCE = 6
MINSIZE = 14
MULTITASK_VALUES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'x', 'y', 'z']
MULTITASK_SIZE = len(MULTITASK_VALUES)

def readData(dataType):
    out = []
    path = "../data/fromWeb/"+dataType+'/'
    ls = os.listdir(path)
    ls = [i for i in ls if i.find(".html")>-1]
    for fn in ls:
        with open(path+fn,'r') as f:
            html = f.read().strip()
        out+= htmlToList(html)
    return out
        
def htmlToList(html):
    out = []
    patt = re.compile("\d{0,4}\)&nbsp; &nbsp;(.+?)&nbsp;&nbsp;&nbsp;<b>answer:</b>(.+?)<br><br><br><br><br><br></td>")
    matches = re.findall(patt,html)
    for match in matches:
        out.append([match[0].strip(),match[1].strip()])
    return out
    
def randomizeVars(prob):
    rep = np.random.choice(['x','y','z'])
    prob = [x.replace('x',rep) for x in prob]
    if np.random.rand()>0.5:
        prob[0] = prob[0].replace(' ','')
    return prob

def problemToImage(prob):
    font = np.random.choice(["Sans","Arial"])
    size = int(np.floor(np.random.rand()*SIZEVARIANCE)+MINSIZE)
    heightVar = MAXHEIGHT-size-1
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, MAXWIDTH, MAXHEIGHT)
    ctx = cairo.Context (surface)
    ctx.set_source_rgb(1,1,1)
    ctx.rectangle(0, 0, MAXWIDTH, MAXHEIGHT)
    ctx.fill()
    ctx.select_font_face(font)
    ctx.set_font_size(size)
    x,y,w,h,dx,dy = ctx.text_extents(prob)
    xPosition = int(np.floor(np.random.rand()*(MAXWIDTH-w)))+1
    yPosition = size+int(np.floor(np.random.rand()*heightVar))+1
    xPosition = np.max([np.min([xPosition,MAXWIDTH-w]),1])
    yPosition = np.max([np.min([yPosition,MAXHEIGHT-h/2]),1])
    ctx.move_to(xPosition,yPosition)
    ctx.set_source_rgb(0,0,0)
    ctx.show_text(prob)
    ctx.stroke()
#    surface.write_to_png('../hello_world.png')
    image = np.frombuffer(surface.get_data(),np.uint8)
    newimage = np.zeros((MAXHEIGHT,MAXWIDTH,3))
    for channel in range(0,3):
        newimage[:,:,channel] = image[channel::4].reshape(MAXHEIGHT,MAXWIDTH)
    newimage /= 255.
#    print(image.shape)
#    print(image)
#    image /= 255
    return newimage
    
def getAnswer(prob):
    return np.float(prob[prob.find("=")+1:].strip())
    
def getGlyphCount(prob):
    r = np.zeros(MULTITASK_SIZE)
    for i in prob:
        if i in MULTITASK_VALUES:
            r[MULTITASK_VALUES.index(i)] += 1
    return r
    
def getTrivialProblem():
    i = int(np.random.randint(0,20))
    return ("x = "+str(i), "x = "+str(i))

def getFullMatrix(dataType,trivialSupplement=1000):
    if dataType.lower().find("test") > -1:
        seedReset = np.random.randint(0,10000)
        np.random.seed(0)
    problems = [randomizeVars(i) for i in readData(dataType)]
    if dataType in ("simple","simpleTest"):
        problems += [randomizeVars(getTrivialProblem()) for i in range(0,trivialSupplement)]
    X = np.zeros((len(problems),MAXHEIGHT,MAXWIDTH,3))
    y = np.zeros((len(problems),1))
    y2 = np.zeros((len(problems),MULTITASK_SIZE))
    for n,problem in enumerate(problems):
        image = problemToImage(problem[0])
        X[n,:,:,:] = image
        y[n] = getAnswer(problem[1])
        y2[n] = getGlyphCount(problem[0])
    np.random.seed(seedReset)
    return X, [y,y2]