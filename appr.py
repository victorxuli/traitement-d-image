# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 13:54:13 2018

@author: victo
"""


from PIL import Image
import numpy as np
import scipy.fftpack as sc
import scipy as sp
import skimage.io as skio
from skimage import exposure as ske
from skimage import color
import matplotlib.pyplot as plt
import skimage


plt.close("all")
#加载，显示图片 灰度图片
I= skio.imread('D:\image\senlin2.jpg');
I = skimage.color.rgb2gray(I)
plt.figure("Original")
plt.title("Original")
plt.imshow(I,cmap = 'gray')
plt.axis('off')

#图像格式转换
skio.imsave('D:\image\cameraman1.jpg',I)

#计算直方图
img=np.array(Image.open('D:\image\cameraman.tif').convert('L'))
plt.figure("cameraman")
arr=img.flatten()
plt.hist(arr, bins=50, facecolor='gray', alpha=0.75)
#n, bins, patches = 
plt.title("histogramme de l'image original")

#伽马校正 gamma = 0.5
gamma_corrected1 = ske.adjust_gamma(I, 0.5);
plt.figure("gamma adjust 0.5")
plt.title("gamma adjust 0.5")
plt.imshow(gamma_corrected1,cmap = 'gray')
#gamma = 2
gamma_corrected2 = ske.adjust_gamma(I, 2);
plt.figure("gamma adjust 2")
plt.title("gamma adjust 2")
plt.imshow(gamma_corrected2,cmap = 'gray')
plt.axis('off')

#图像均衡化
I_eq = ske.equalize_hist(I)
plt.figure("equalisation")
plt.title("equalisation")
arr = I_eq.flatten()
n1 = plt.hist(arr, bins=50, facecolor='gray', alpha=0.75)
plt.figure("equalisation")  # 显示图像均衡化
plt.imshow(I_eq,cmap='gray')
plt.title("image equalisation")
plt.axis('off')

# ajustement linéaire de l’histogramme 线性调整 10% - 90%
Imax = np.percentile(I,90)
Imax = int(Imax)
Imin = np.percentile(I,10)  
Imin = int(Imin)
I_adj = ske.rescale_intensity(I,in_range=(Imin,Imax))
plt.figure("adjust lineaire")
plt.imshow(I_adj,cmap='gray')
plt.title("image adjust lineaire")

#加载彩色图片 sport23.jpg
I_color= skio.imread('D:\image\sport23.jpg');
plt.figure("image color Original")
plt.title("image color Original")
plt.imshow(I_color)
plt.axis('off')

#颜色空间及其转换 rgb->lab
I_lab = color.rgb2lab(I_color);
plt.figure("image rgb2lab")
plt.imshow(I_lab)
plt.title("image rgb2lab")
# Lab颜色空间中的L分量用于表示像素的亮度，取值范围是[0,100],表示从纯黑到纯白；
# a表示从红色到绿色的范围，    ；
# b表示从黄色到蓝色的范围，取值范围是[127,-128]。
# Lab颜色空间比计算机显示器、打印机甚至比人类视觉的色域都要大 
# RGB无法直接转换成LAB，需要先转换成XYZ再转换成LAB，即：RGB——XYZ——LAB

# 从图片中提取滑翔伞帆
plt.figure("image rgb2lab plan 1 ")
plt.imshow(I_lab[:,:,0],cmap = "gray")
plt.title("image rgb2lab plan 1")

plt.figure("image rgb2lab plan 2 ")
plt.imshow(I_lab[:,:,1],cmap = "gray")
plt.title("image rgb2lab plan 2")

plt.figure("image rgb2lab plan 3 ")
plt.imshow(I_lab[:,:,2],cmap = "gray")
plt.title("image rgb2lab plan 3")



# 图像二值化  lab 第二个平面二值化
rows,cols=I_lab[:,:,1].shape
#masque = I_lab[:,:,1];
I_E = np.ones([rows,cols])
masque1 = np.multiply(I_lab[:,:,1],I_E)
for i in range(rows):
    for j in range(cols):
        if (masque1[i,j]<=15):
            masque1[i,j]=127
        else:
            masque1[i,j]=-127
            
plt.figure("image binarisé le plan 2")            
plt.imshow(masque1,cmap = "gray")
plt.title("image binarisé le plan 2")
#lab图像第一平面加窗
plan_1_masque = np.multiply(I_lab[:,:,0],masque1)
plt.figure("image apres masque le plan 1")            
plt.imshow(plan_1_masque,cmap = "gray")
plt.title("image apres masque le plan 1")
##lab图像第二平面加窗
plan_2_masque = np.multiply(I_lab[:,:,1],masque1)
plt.figure("image apres masque le plan 2")            
plt.imshow(plan_2_masque,cmap = "gray")
plt.title("image apres masque le plan 2")
##lab图像第三平面加窗
plan_3_masque = np.multiply(I_lab[:,:,2],masque1)
plt.figure("image apres masque le plan 3")            
plt.imshow(plan_3_masque,cmap = "gray")
plt.title("image apres masque le plan 3")
##RVB图像第一平面rouge加窗
planRVB_1_masque = np.multiply(I_color[:,:,0],masque1)
plt.figure("image RVB apres masque le plan 1")            
plt.imshow(planRVB_1_masque,cmap = "gray")
plt.title("image RVB apres masque le plan 1")
##RVB图像第二平面vert加窗
planRVB_2_masque = np.multiply(I_color[:,:,1],masque1)
plt.figure("image RVB apres masque le plan 2")            
plt.imshow(planRVB_2_masque,cmap = "gray")
plt.title("image RVB apres masque le plan 2")
##RVB图像第三平面bleu加窗
planRVB_3_masque = np.multiply(I_color[:,:,2],masque1)
plt.figure("image RVB apres masque le plan 3")            
plt.imshow(planRVB_2_masque,cmap = "gray")
plt.title("image RVB apres masque le plan 3")

#
I_lab[:,:,0] = np.multiply(I_lab[:,:,0],masque1)
I_lab[:,:,1] = np.multiply(I_lab[:,:,1],masque1)
I_lab[:,:,2] = np.multiply(I_lab[:,:,2],masque1)
plt.figure("image RVB apres masque")            
plt.imshow(I_lab,cmap = "gray")
plt.title("image RVB apres masque")


##图像公式
[x,y]=np.ogrid[-192:193,-192:193]
d=np.sqrt(x*x+y*y);
f1=0.02;
f2=0.2;
A=0.5+0.25*np.cos(2*np.pi*f1*d)+0.25*np.cos(2*np.pi*f2*d);
plt.figure("image o")
plt.imshow(A,cmap = "gray")
#
#
#图片采样（每间隔4采样）
Ise4 = np.copy(A[::4,::4])
plt.figure("image echantillone 1/4")
plt.imshow(Ise4,cmap = "gray")
plt.title("image echantillone 1/4")
#
#图片采样（每间隔4采样）
Ise2 = np.copy(A[::2,::2])
plt.figure("image echantillone 1/2")
plt.imshow(Ise2,cmap = "gray")
plt.title("image echantillone 1/2")

##图像傅里叶变换
#TF1 = sc.fft2(A)
#TF2 = sc.fftshift(TF1)
#spectre = np.abs(TF2)
#plt.figure("spectre de A")
#plt.imshow(np.log10(1+spectre),cmap = "gray")
#plt.title("spectre de A")


#图像傅里叶变换
TF1 = sc.fft2(A)
TF2 = sc.fftshift(TF1)
spectre = np.abs(TF2)
spectre1 = np.abs(spectre)
plt.figure("spectre de A")
plt.imshow(np.log10(1+spectre1),cmap = "gray")
plt.title("spectre de A")

#傅里叶变换后图像相位角
phase = sp.angle(TF2)
plt.figure("phase de TF")
plt.imshow(phase,cmap = "gray")
plt.title("phase de TF")


#TF1_bateau = sc.fft2(imag)
#TF2_bateau = sc.fftshift(TF1_bateau)
#spectre_bateau = np.abs(TF2_bateau)
#plt.figure("spectre de bateau")
#plt.imshow(np.log10(1+spectre_bateau),cmap = "gray")
#plt.title("spectre de bateau")

def fft_image(adresse_image,A):
    imag = skio.imread(adresse_image);
    plt.figure(A)
    plt.title(A)
    plt.imshow(imag,cmap = 'gray')
    plt.axis('off')
    TF1_bateau = sc.fft2(imag)
    TF2_bateau = sc.fftshift(TF1_bateau)
    spectre_bateau = np.abs(TF2_bateau)
    plt.figure(adresse_image)
    plt.imshow(np.log10(1+spectre_bateau),cmap = "gray")
    plt.title(adresse_image)

fft_image('D:\image\BATEAU.tif',"BATEAU Original")
fft_image('D:\image\COULOIR.tif',"COULOIR Original")

