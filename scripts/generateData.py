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
import sys
import re

MAXWIDTH = 128
MAXHEIGHT = 128
SIZEVARIANCE = 6
MINSIZE = 14
MULTITASK_VALUES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'x', 'y', 'z']
MULTITASK_SIZE = len(MULTITASK_VALUES)
    
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
    newimage = np.zeros((MAXHEIGHT,MAXWIDTH,1))
    for channel in range(0,1):
        newimage[:,:,channel] = image[channel::4].reshape(MAXHEIGHT,MAXWIDTH)
    newimage /= 255.
    newimage = np.ones_like(newimage) - newimage
    newimage = newimage[:,:,0]
    return newimage

    
def getAnswer(prob):
    return np.float(prob[prob.find("=")+1:].strip())

    
def getGlyphCount(prob):
    r = np.zeros(MULTITASK_SIZE)
    for i in prob:
        if i in MULTITASK_VALUES:
            r[MULTITASK_VALUES.index(i)] += 1
    return r


def readData(counts, reservedProblems):
    out = []
    for key in counts.keys():
        for c in range(0,counts[key]):
            out.append(generateAlgebra(key, reservedProblems))
    return out


def getFullMatrix(dataType, counts, reservedProblems):
    seedReset = np.random.randint(0,10000)
    if (dataType.lower() == "test") or (dataType.lower().find("val") > -1):
        np.random.seed(0)
    origProblems = readData(counts, reservedProblems)
    problems = [randomizeVars(i) for i in origProblems]
    X = np.zeros((len(problems),MAXHEIGHT,MAXWIDTH,1))
    y = np.zeros((len(problems),1))
    y2 = np.zeros((len(problems),MULTITASK_SIZE))
    for n,problem in enumerate(problems):
        image = problemToImage(problem[0])
        X[n,:,:,0] = image
        y[n] = getAnswer(problem[1])
        y2[n] = getGlyphCount(problem[0])
    np.random.seed(seedReset)
    return X, [y,y2], problems


def generateAlgebra(probType, reservedProblems, depth = 0):
    coeff = 0
    rhs = np.random.randint(0,200) - 100
    var = 'x'
    if probType == "trivial":
        coeff = 1
        added = 0
    if probType == "simple":
        while coeff == 0:
            coeff = np.random.randint(0,30) - 15
        added = np.random.randint(0,100) - 50
    prob, ans = solveSimple(var,coeff,added,rhs)
    if prob in reservedProblems:
        print("problem already in reservedProblems: ", prob)
        if depth > 50:
            print("can't find original problem of type: ",probType)
            sys.exit(1)
        return generateAlgebra(probType, reservedProblems, depth=depth+1)

    ansString = var+' = '+str(ans)
    return prob, ansString


def solveSimple(var, coeff, added, rhs, forceRound=True, inc=None):
    if coeff == 1:
        sc = ''
    else:
        sc = str(coeff)
    if np.random.rand() > 0.5:
        if added > 0:
            probString = sc+var+' + '+str(added)+' = '+str(rhs)
        elif added < 0:
            probString = sc+var+' - '+str(abs(added))+' = '+str(rhs)
        else:
            probString = sc+var+' = '+str(rhs)
    else:
        if added == 0:
            probString = sc+var+' = '+str(rhs)
        else:
            probString = str(added)+' + '+sc+var+' = '+str(rhs)
    newrhs = rhs - added
    ans = newrhs*1. / coeff
#    print(probString, ans)
    if forceRound:
        if inc == None:
            inc = np.random.choice([-1,1])
        if ans == int(ans):
            return probString, ans
        else:
            return solveSimple(var, coeff, added+inc, rhs, inc=inc)
    else:
        return probString, ans
        