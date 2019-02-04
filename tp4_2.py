# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 16:12:40 2018

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
imag = skio.imread(r'D:\image\cellules2b.JPG')
plt.figure("image original ")
plt.title("image original ")
plt.imshow(imag,cmap = 'gray')
plt.axis('off')

## 彩色图像转灰色图像
imag = color.rgb2gray(imag)
plt.figure("image gry ")
plt.title("image gry ")
plt.imshow(imag,cmap = 'gray')
plt.axis('off')


## 图像底帽操作 将原图像减去它的闭运算值，返回比结构化元素小的黑点，且将这些黑点反色。 
dst10=sm.black_tophat(imag,sm.disk(10))  # 边长为10 获得所有细胞
plt.figure("image black_tophat 15")
plt.title("image black_tophat 15")
plt.imshow(dst10,cmap = 'gray')
plt.axis('off')

## 
dst11=sm.black_tophat(imag,sm.disk(2)) #获得小的细胞
plt.figure("image black_tophat 2")
plt.title("image black_tophat 2")
plt.imshow(dst11,cmap = 'gray')
plt.axis('off')

#ass = fl.threshold_otsu(dst10)
#dst12 = dst10<=ass

#rows,cols=dst12.shape
#for i in range(rows):
#    for j in range(cols):
#        if (dst12[i,j]==0):
#            dst12[i,j]=1
#        else:
#            dst12[i,j]=0
#dst12 = sm.opening(dst12,sm.disk(2))
#plt.figure("image binarisation auto")
#plt.title("image binarisation auto")
#plt.imshow(dst12,cmap = 'gray')
#plt.axis('off')

#
Imax = np.percentile(dst10,82)
rows,cols=dst10.shape
for i in range(rows):
    for j in range(cols):
        if (dst10[i,j]>=Imax):
            dst10[i,j]=1
        else:
            dst10[i,j]=0
dst10 = sm.opening(dst10,sm.disk(2))
dst10 = clear_border(dst10)
plt.figure("2")
plt.title("2")
plt.imshow(dst10,cmap = 'gray')
plt.axis('off')

h = sm.label(dst10)
H = ske.histogram(h)

tmp1 = len(H[1])
tmp2 = len(H[0])
x = H[1][1:tmp1]
y = H[0][1:tmp2]
plt.figure('hist')
plt.stem(x,y)
plt.title('histogramme big cellule')



#ass1 = fl.threshold_otsu(dst11)
#dst13 = dst11<=ass1;
Imax1 = np.percentile(dst11,90)
rows,cols=dst10.shape
for i in range(rows):
    for j in range(cols):
        if (dst11[i,j]>=Imax1):
            dst11[i,j]=1
        else:
            dst11[i,j]=0
dst11 = sm.opening(dst11,sm.disk(1))
dst11 = clear_border(dst11)
plt.figure("xiao xibao ")
plt.title("xiao xibao")
plt.imshow(dst11,cmap = 'gray')
plt.axis('off') 

h1 = sm.label(dst11)
H1 = ske.histogram(h1)

tmp3 = len(H1[1])
tmp4 = len(H1[0])
x1 = H1[1][1:tmp3]
y1 = H1[0][1:tmp4]
plt.figure('hist little cellule')
plt.stem(x1,y1)
plt.title('histogramme little cellule')



imag = skio.imread(r'D:\image\sport23.JPG')
plt.figure("image original")
plt.title("image original")
plt.imshow(imag,cmap = 'gray')
plt.axis('off')
for m in range(100,200):
    for n in range(100,200):
        if m%10==0 and n%10 ==0:
            for i in range(0,10):
                for j in range(0,10):
                    (b,r,g) = imag[m][n]
                    imag[i+m][j+n] = (b,r,g)
plt.figure("image masaike 10")
plt.title("image masaike 10")
plt.imshow(imag,cmap = 'gray')
plt.axis('off')       

imag = skio.imread(r'D:\image\sport23.JPG')
plt.figure("image original")
plt.title("image original")
plt.imshow(imag,cmap = 'gray')
plt.axis('off')
for m in range(100,200):
    for n in range(100,200):
        if m%20==0 and n%20 ==0:
            for i in range(0,20):
                for j in range(0,20):
                    (b,r,g) = imag[m][n]
                    imag[i+m][j+n] = (b,r,g)
plt.figure("image masaike 20")
plt.title("image masaike 20")
plt.imshow(imag,cmap = 'gray')
plt.axis('off')    


#skimage.img_as_float32

imag = skio.imread(r'D:\image\雀斑.JPG',1)
imag = skimage.img_as_float32(imag)
plt.figure("image original")
plt.title("image original")
plt.imshow(imag,cmap = 'gray')
plt.axis('off')
#imag = color.rgb2gray(imag)
#高斯噪声图片过双边滤波器
selem = disk(20)
#fl.rank.sum_bilateral
#imag_bila = fl.rank.sum_bilateral(imag,selem = selem, out=None, mask=None, shift_x=False, shift_y=False, s0=1, s1=2)
imag = cv2.bilateralFilter(imag,10,20,20)
plt.figure("image CARRE1 apres filtrage de mean_bilateral ")
plt.title("image CARRE1 apres filtrage de mean_bilateral")
plt.imshow(imag,cmap = 'gray')
plt.axis('off')