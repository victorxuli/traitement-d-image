# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 13:51:54 2018

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
import skimage.measure as skim
from skimage.morphology import disk
import skimage.restoration as re
import skimage 
import cv2
from skimage.feature import canny
plt.close("all")

#imag = skio.imread('D:\image\CARRE1.tif')
imag = skio.imread('D:\image\BATEAU.tif')
imag_float = img_as_float(imag)
imag_gaussian = ut.random_noise(imag, mode='gaussian', seed=None, clip=True,mean = 0,var = 0.05)
imag_pepper = ut.random_noise(imag, mode='pepper',seed=None, clip=True)
plt.figure("image original")
plt.imshow(imag,cmap='gray')
plt.axis('off')


##### Filtre FIR 
# filtrage sobel 
I_PH = fl.sobel(imag)
plt.figure("image passe filtre sobel ")
plt.title("image passe filtre sobel ")
plt.imshow(I_PH,cmap = 'gray')
plt.axis('off')

# filtrage prewitt 
I_PH = fl.prewitt(imag)
plt.figure("image passe filtre prewitt ")
plt.title("image passe filtre prewitt ")
plt.imshow(I_PH,cmap = 'gray')
plt.axis('off')

# filtrage robert
I_PH = fl.roberts(imag)
plt.figure("imag passe filtre roberts ")
plt.title("imag passe filtre roberts ")
plt.imshow(I_PH,cmap = 'gray')
plt.axis('off')

# filtrage canny
##Canny算法的步骤:
#降噪:
#任何边缘检测算法都不可能在未经处理的原始数据上很好地处理，
#所以第一步是对原始数据与高斯平滑模板作卷积，得到的图像与原始图像相比有些轻微的模糊（blurred）。
#这样，单独的一个像素噪声在经过高斯平滑的图像上变得几乎没有影响

#寻找图像中的亮度梯度:
#图像中的边缘可能会指向不同的方向，所以Canny算法使用4个mask检测水平、垂直以及对角线方向的边缘。
#原始图像与每个mask所作的卷积都存储起来。对于每个点我们都标识在这个点上的最大值以及生成的边缘的方向。
#这样我们就从原始图像生成了图像中每个点亮度梯度图以及亮度梯度的方向。

## 较高的亮度梯度比较有可能是边缘，但是没有一个确切的值来限定多大的亮度梯度是边缘多大又不是，所以Canny使用了滞后阈值。
#滞后阈值需要两个阈值——高阈值与低阈值。假设图像中的重要边缘都是连续的曲线，这样我们就可以跟踪给定曲线中模糊的部分，
#并且避免将没有组成曲线的噪声像素当成边缘。所以我们从一个较大的阈值开始，这将标识出我们比较确信的真实边缘，使用前面导出的方向信息，
#我们从这些真正的边缘开始在图像中跟踪整个的边缘。在跟踪的时候，我们使用一个较小的阈值，
#这样就可以跟踪曲线的模糊部分直到我们回到起点


I_PH = canny(imag)
plt.figure("imag passe filtre canny ")
plt.title("imag passe filtre canny ")
plt.imshow(I_PH,cmap = 'gray')
plt.axis('off')



#寻找图像阈值

seuil = fl.threshold_mean(imag)
seuil = int(seuil)
rows,cols=imag.shape
for i in range(rows):
    for j in range(cols):
        if (imag[i,j]>=seuil):
            imag[i,j]=255
        else:
            imag[i,j]=0
plt.figure("image bilanisation ")
plt.title("image bilanisation ")
plt.imshow(imag,cmap = 'gray')
plt.axis('off')


seuil = fl.threshold_mean(imag_gaussian)
rows,cols=imag_gaussian.shape
for i in range(rows):
    for j in range(cols):
        if (imag_gaussian[i,j]>=seuil):
            imag_gaussian[i,j]=1
        else:
            imag_gaussian[i,j]=0
plt.figure("image bruite bilanisation ")
plt.title("image bruite bilanisation ")
plt.imshow(imag_gaussian,cmap = 'gray')
plt.axis('off')
#阈值高通滤波器对噪声图像效果不好

## Filtre IIR
# filtre canny
#I_PH = fl.(imag)
#plt.figure("image passe filtre sobel ")
#plt.title("image passe filtre sobel ")
#plt.imshow(I_PH,cmap = 'gray')
#plt.axis('off')



###########
I_PH = fl.sobel(imag_pepper)
plt.figure("imag_pepper passe filtre sobel ")
plt.title("imag_pepper passe filtre sobel ")
plt.imshow(I_PH,cmap = 'gray')
plt.axis('off')

I_PH = fl.prewitt(imag_pepper)
plt.figure("imag_pepper passe filtre prewitt ")
plt.title("imag_pepper passe filtre prewitt ")
plt.imshow(I_PH,cmap = 'gray')
plt.axis('off')

I_PH = fl.roberts(imag_pepper)
plt.figure("imag_pepper passe filtre roberts ")
plt.title("imag_pepper passe filtre roberts ")
plt.imshow(I_PH,cmap = 'gray')
plt.axis('off')

I_PH = canny(imag_pepper)
plt.figure("imag_pepper passe filtre canny ")
plt.title("imag_pepper passe filtre canny ")
plt.imshow(I_PH,cmap = 'gray')
plt.axis('off')

###########
I_PH = fl.sobel(imag_gaussian)
plt.figure("imag_gaussian passe filtre sobel ")
plt.title("imag_gaussian passe filtre sobel ")
plt.imshow(I_PH,cmap = 'gray')
plt.axis('off')

I_PH = fl.prewitt(imag_gaussian)
plt.figure("imag_gaussian passe filtre prewitt ")
plt.title("imag_gaussian passe filtre prewitt ")
plt.imshow(I_PH,cmap = 'gray')
plt.axis('off')

I_PH = fl.roberts(imag_gaussian)
plt.figure("imag_gaussian passe filtre roberts ")
plt.title("imag_gaussian passe filtre roberts ")
plt.imshow(I_PH,cmap = 'gray')
plt.axis('off')

I_PH = canny(imag_gaussian)
plt.figure("imag_gaussian passe filtre canny ")
plt.title("imag_gaussian passe filtre canny ")
plt.imshow(I_PH,cmap = 'gray')
plt.axis('off')


##########
laplacian = fl.edges.laplacian(2,(rows,cols))
