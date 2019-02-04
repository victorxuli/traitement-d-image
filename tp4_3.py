# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 17:36:12 2018

@author: victo
"""

from skimage import util as ut
from PIL import Image
import numpy as np
import scipy.fftpack as sc
import scipy as sp
import skimage.io as skio
from skimage import exposure as ske
from skimage import color
import matplotlib.pyplot as plt
from skimage import filters as fl
from skimage import img_as_float as img_as_float
from skimage.segmentation import clear_border
import skimage.measure as skim
from skimage.morphology import disk
import skimage.restoration as re
import skimage 
import skimage.morphology as sm
import cv2

plt.close("all")

#imag = skio.imread(r'D:\image\xxx.JPG')
imag = skio.imread(r'D:\image\cameraman.tif')
plt.figure("image originale ")
plt.title("image originale ")
plt.imshow(imag,cmap = 'gray')
plt.axis('off')
#imag = img_as_float(imag)

imag = color.rgb2gray(imag)
plt.figure("image originale gray ")
plt.title("image originale gray ")
plt.imshow(imag,cmap = 'gray')
plt.axis('off')


## gradient  膨胀 - 腐蚀
#dst1 = sm.dilation(imag,sm.disk(6)) - sm.erosion(imag,sm.disk(6))
#plt.figure("image gradient ")
#plt.title("image gradient ")
#plt.imshow(dst1,cmap = 'gray')
#plt.axis('off')
#
#Imax1 = np.percentile(dst1,88)
#rows,cols=dst1.shape
#for i in range(rows):
#    for j in range(cols):
#        if (dst1[i,j]>=Imax1):
#            dst1[i,j]=1
#        else:
#            dst1[i,j]=0
#dst1 = clear_border(dst1)
#plt.figure("image Binarisation ")
#plt.title("image Binarisation ")
#plt.imshow(dst1,cmap = 'gray')
#plt.axis('off')

# laplacien
#dst1 = (sm.dilation(imag,sm.disk(6)) + sm.erosion(imag,sm.disk(6))-imag-imag)/2
#plt.figure("image laplacien ")
#plt.title("image laplacien ")
#plt.imshow(dst1,cmap = 'gray')
#plt.axis('off')
#
#Imax1 = np.percentile(dst1,94)
#rows,cols=dst1.shape
#for i in range(rows):
#    for j in range(cols):
#        if (dst1[i,j]>=Imax1):
#            dst1[i,j]=1
#        else:
#            dst1[i,j]=0
#dst1 = clear_border(dst1)
#plt.figure("image Binarisation ")
#plt.title("image Binarisation ")
#plt.imshow(dst1,cmap = 'gray')
#plt.axis('off')



#dst1 = imag + sm.white_tophat(imag,sm.disk(1)) - sm.black_tophat(imag,sm.disk(1))
#plt.figure("image top hat ")
#plt.title("image top hat ")
#plt.imshow(dst1,cmap = 'gray')
#plt.axis('off')


#边缘检测 原图-腐蚀
#dst1 = imag - sm.erosion(imag,sm.disk(2))
#plt.figure("image top hat ")
#plt.title("image top hat ")
#plt.imshow(dst1,cmap = 'gray')
#plt.axis('off')
#Imax1 = np.percentile(dst1,94.41)
#rows,cols=dst1.shape
#for i in range(rows):
#    for j in range(cols):
#        if (dst1[i,j]>=Imax1):
#            dst1[i,j]=1
#        else:
#            dst1[i,j]=0
#dst1 = clear_border(dst1)
#plt.figure("image Binarisation ")
#plt.title("image Binarisation ")
#plt.imshow(dst1,cmap = 'gray')
#plt.axis('off')


dst1 = sm.white_tophat(imag,sm.disk(50))
plt.figure("image top hat ")
plt.title("image top hat ")
plt.imshow(dst1,cmap = 'gray')
plt.axis('off')

Imax1 = np.percentile(dst1,95)
rows,cols=dst1.shape
for i in range(rows):
    for j in range(cols):
        if (dst1[i,j]>=Imax1):
            dst1[i,j]=1
        else:
            dst1[i,j]=0
dst1 = clear_border(dst1)
plt.figure("image Binarisation ")
plt.title("image Binarisation ")
plt.imshow(dst1,cmap = 'gray')
plt.axis('off')

h = sm.label(dst1)
H = ske.histogram(h)
x = H[1][1:len(H[1])]
y = H[0][1:len(H[0])]
plt.figure('hist')
plt.stem(x,y)
plt.title('histogramme')
yMax = max(y)
surface = (rows*cols)
pourcentage1 = yMax/surface
pourcentage2 = sum(y)/surface


