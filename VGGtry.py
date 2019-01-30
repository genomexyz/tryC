#!/usr/bin/python

import numpy as np
import keras
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from PIL import Image as pil_image

#setting
width = 224
height = 224
imgdir = 'img'
outfile = 'result.txt'

#load all image
allimgname = sorted(os.listdir(imgdir))
imgnumber = len(allimgname)
allimg = []
for i in xrange(imgnumber):
	img = pil_image.open(imgdir+'/'+allimgname[i])
	img = img.resize((width, height))
	dataimg = np.array(img)
#in PNG image, 4-th channel is transparency channel
#so we ignore it in PNG case
	allimg.append(dataimg[:,:,:3])
dataallimg = np.float64(np.asarray(allimg))
#preprocessing, adequate our range number in img to match model, in this case VGG
dataallimg = preprocess_input(dataallimg)

#build model
mod = VGG16()

#make prediction
pred = mod.predict(dataallimg)

#convert the probabilities to class labels
predlabel = decode_predictions(pred)

out = open(outfile, 'w')
for i in xrange(len(predlabel)):
	res = predlabel[i][0]
	out.write(allimgname[i]+" -> "+res[1]+" ("+str(res[2]*100)+")\n")

#get the highest prob
#predlabel = predlabel[0][0]

#result
#print(predlabel[1]+" prob-> ("+str(predlabel[2]*100)+")")
