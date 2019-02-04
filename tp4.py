# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 14:10:42 2018

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
imag = skio.imread('D:\image\code.bmp')
plt.figure("image original ")
plt.title("image original ")
plt.imshow(imag,cmap = 'gray')
plt.axis('off')

## mdian
seuil1 = np.median(imag)
rows,cols=imag.shape
for i in range(rows):
    for j in range(cols):
        if (imag[i,j]>=seuil1):
            imag[i,j]=255
        else:
            imag[i,j]=0
plt.figure("image bilanisation par median")
plt.title("image bilanisation par median ")
plt.imshow(imag,cmap = 'gray')
plt.axis('off')

## mean
imag = skio.imread('D:\image\code.bmp')
seuil2 = np.mean(imag)
rows,cols=imag.shape
for i in range(rows):
    for j in range(cols):
        if (imag[i,j]>=seuil2):
            imag[i,j]=255
        else:
            imag[i,j]=0
plt.figure("image bilanisation par mean")
plt.title("image bilanisation par mean ")
plt.imshow(imag,cmap = 'gray')
plt.axis('off')

##图像膨胀滤波 图像变亮 粗化区域 亮特征增大 暗特征减小 dilatation 
imag = skio.imread('D:\image\code.bmp')
dst1=sm.dilation(imag,sm.square(3))  #用边长为3的正方形滤波器进行膨胀滤波
plt.figure("image dilation 3")
plt.title("image dilation 3")
plt.imshow(dst1,cmap = 'gray')
plt.axis('off')


dst2=sm.dilation(imag,sm.square(11))  #用边长为11的正方形滤波器进行膨胀滤波
plt.figure("image dilation 11")
plt.title("image dilation 11")
plt.imshow(dst2,cmap = 'gray')
plt.axis('off')

##图像腐蚀滤波 图像变暗 细化区域 亮特征减小 暗特征增加  erosion 
dst5=sm.erosion(imag,sm.square(5))  #用边长为5的正方形滤波器进行腐蚀滤波
plt.figure("image erosion 5")
plt.title("image erosion 5")
plt.imshow(dst5,cmap = 'gray')
plt.axis('off')
dst6=sm.erosion(imag,sm.square(25))  #用边长为25的正方形滤波器进行腐蚀滤波
plt.figure("image erosion 25")
plt.title("image erosion 25")
plt.imshow(dst6,cmap = 'gray')
plt.axis('off')

##图像开运算 先腐蚀再膨胀 小亮点消失 原图中大片黑色区域中的亮点消失 可以消除小物体或小斑块
dst3=sm.opening(imag,sm.disk(9))  #用边长为9的圆形滤波器进行膨胀滤波
plt.figure("image openning 9")
plt.title("image openning 9")
plt.imshow(dst3,cmap = 'gray')
plt.axis('off')

dst4=sm.opening(imag,sm.disk(3))  #用边长为3的圆形滤波器进行膨胀滤波
plt.figure("image openning 3")
plt.title("image openning 3")
plt.imshow(dst4,cmap = 'gray')
plt.axis('off')

##图像闭运算 先膨胀再腐蚀，可用来填充孔洞。
dst7=sm.closing(imag,sm.disk(9))  #用边长为9的圆形滤波器进行膨胀滤波
plt.figure("image close 9")
plt.title("image close 9")
plt.imshow(dst7,cmap = 'gray')
plt.axis('off')

dst8=sm.closing(imag,sm.disk(3))  #用边长为3的圆形滤波器进行膨胀滤波
plt.figure("image close 3")
plt.title("image close 3")
plt.imshow(dst8,cmap = 'gray')
plt.axis('off')

##图像白帽（white-tophat) 将原图像减去它的开运算值，返回比结构化元素小的白点
dst9=sm.white_tophat(imag,sm.square(25))
plt.figure("image white_tophat 25")
plt.title("image white_tophat 25")
plt.imshow(dst9,cmap = 'gray')
plt.axis('off')

##图像黑帽（black-tophat) 将原图像减去它的闭运算值，返回比结构化元素小的黑点，且将这些黑点反色。 bottom-Hat
dst10=sm.black_tophat(imag,sm.square(23))
dst10 = clear_border(dst10)
plt.figure("image black_tophat 27")
plt.title("image black_tophat 27")
plt.imshow(dst10,cmap = 'gray')
plt.axis('off')
dst10=sm.opening(dst10,sm.disk(6))

#dst10=sm.opening(dst10,sm.square(3))
#dst10=sm.closing(dst10,sm.disk(6))
#dst10=sm.erosion(dst10,sm.square(6))
dst10=sm.dilation(dst10,sm.square(12))
#dst10=sm.erosion(dst10,sm.square(4))


#Imax1 = np.percentile(dst10,60)
#rows,cols=dst10.shape
#for i in range(rows):
#    for j in range(cols):
#        if (dst10[i,j]<=Imax1):
#            dst10[i,j]=0
dst10 = clear_border(dst10)            
plt.figure("image bilanisation par mean_final")
plt.title("image bilanisation par mean_final ")
plt.imshow(dst10,cmap = 'gray')
plt.axis('off')

#dst10=sm.opening(dst10,sm.disk(3)) 
#dst10=sm.dilation(dst10,sm.square(3)) 
     
#plt.figure("image black_tophat asd 25")
#plt.title("image black_tophat das 25")
#plt.imshow(dst10,cmap = 'gray')
#plt.axis('off')
#seuil_f = np.mean(dst10)
ass = fl.threshold_otsu(dst10)
dst10 = dst10<=ass
#Imax = np.percentile(dst10,87)
rows,cols=dst10.shape
for i in range(rows):
    for j in range(cols):
        if (dst10[i,j]==True):
            dst10[i,j]=False
        else:
            dst10[i,j]=True
#dst10 = clear_border(dst10)
dst10=sm.erosion(dst10,sm.square(5))
plt.figure("image bilanisation ")
plt.title("image bilanisation")
plt.imshow(dst10,cmap = 'gray')
plt.axis('off')



