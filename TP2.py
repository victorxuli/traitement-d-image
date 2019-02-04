# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 13:45:58 2018

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

plt.close("all")
#图像加噪声(高斯噪声)
imag = skio.imread('D:\image\cameraman.tif');
#imag = skio.imread('D:\image\chasse.tif');
#imag = skio.imread('D:\image\cameraman.tif');
imag_float = img_as_float(imag)
imag_gaussian = ut.random_noise(imag, mode='gaussian', seed=None, clip=True)
plt.figure("image CARRE1 bruite avec gaussien")
plt.title("image CARRE1 bruite avec gaussien")
plt.imshow(imag_gaussian,cmap = 'gray')
plt.axis('off')

#
imag = skio.imread('D:\image\CARRE1.tif');
imag_pepper = ut.random_noise(imag, mode='pepper', seed=None, clip=True)
plt.figure("image CARRE1 bruite avec pepper")
plt.title("image CARRE1 bruite avec pepper")
plt.imshow(imag_pepper,cmap = 'gray')
plt.axis('off')

###
#imag = skio.imread('D:\image\BATEAU.tif');
#imag_float = img_as_float(imag)
#imag_gaussian = ut.random_noise(imag, mode='gaussian', seed=None, clip=True)
#plt.figure("image CARRE1 BATEAU avec gaussien")
#plt.title("image CARRE1 BATEAU avec gaussien")
#plt.imshow(imag_gaussian,cmap = 'gray')
#plt.axis('off')
#
##
#imag = skio.imread('D:\image\BATEAU.tif');
#imag_pepper = ut.random_noise(imag, mode='pepper', seed=None, clip=True)
#plt.figure("image BATEAU bruite avec pepper")
#plt.title("image BATEAU bruite avec pepper")
#plt.imshow(imag_pepper,cmap = 'gray')
#plt.axis('off')

#高斯滤波器
#I_apres_filtre_gau = fl.gaussian(imag_gaussian,sigma=1,output=None, mode='nearest', cval=0, multichannel=None, preserve_range=False, truncate=4.0)
I_apres_filtre_gau = fl.gaussian(imag_gaussian)
plt.figure("image CARRE1 apres filtrage gaussian ")
plt.title("image CARRE1 apres filtrage gaussian")
plt.imshow(I_apres_filtre_gau,cmap = 'gray')
plt.axis('off')

I_apres_filtre_gau_1 = fl.gaussian(imag_pepper)
plt.figure("image CARRE1 pepper apres filtrage gaussian ")
plt.title("image CARRE1 pepper apres filtrage gaussian")
plt.imshow(I_apres_filtre_gau_1,cmap = 'gray')
plt.axis('off')

#中值滤波器
I_apres_filtre_median = fl.median(imag_gaussian, selem=None)
plt.figure("image CARRE1  apres filtrage de median ")
plt.title("image CARRE1  apres filtrage de median")
plt.imshow(I_apres_filtre_median,cmap = 'gray')
plt.axis('off')

I_apres_filtre_median_1 = fl.median(imag_pepper, selem=None)
plt.figure("image CARRE1 pepper apres filtrage de median ")
plt.title("image CARRE1 pepper apres filtrage de median")
plt.imshow(I_apres_filtre_median_1,cmap = 'gray')
plt.axis('off')

#原图与过了噪声图片信噪比
PSNR_bruit_Gaussian = skim.compare_psnr(imag_float,imag_gaussian,data_range=None, dynamic_range=None)
PSNR_bruit_pepper = skim.compare_psnr(imag_float,imag_pepper,data_range=None, dynamic_range=None)
## PSNR_bruit_Gaussian = 22.9 // PSNR_bruit_pepper = 13.5


#原图与过了滤波器的图像信噪比
PSNR_Gaussian = skim.compare_psnr(imag_float,I_apres_filtre_gau,data_range=None, dynamic_range=None)
PSNR_median = skim.compare_psnr(imag,I_apres_filtre_median,data_range=None, dynamic_range=None)
## PSNR_Gaussian = 22.74 // PSNR_median = 28.79

#高斯噪声图片过双边滤波器
selem = disk(20)
imag_bila = fl.rank.mean_bilateral(imag_gaussian,selem = selem, out=None, mask=None, shift_x=False, shift_y=False, s0=2, s1=2)
plt.figure("image CARRE1 apres filtrage de mean_bilateral ")
plt.title("image CARRE1 apres filtrage de mean_bilateral")
plt.imshow(imag_bila,cmap = 'gray')
plt.axis('off')

#椒盐噪声图片过双边滤波器
imag_bila1 = fl.rank.mean_bilateral(imag_pepper,selem = selem, out=None, mask=None, shift_x=False, shift_y=False, s0=10, s1=10)
plt.figure("image CARRE1 apres filtrage de mean_bilateral pepper ")
plt.title("image CARRE1 apres filtrage de mean_bilateral pepper ")
plt.imshow(imag_bila1,cmap = 'gray')
plt.axis('off')

#原图与过了双边滤波器的高斯噪声图像/椒盐噪声图像 信噪比
PSNR_Gaussian_bila = skim.compare_psnr(imag,imag_bila,data_range=None, dynamic_range=None)
PSNR_pepper_bila = skim.compare_psnr(imag,imag_bila1,data_range=None, dynamic_range=None)
# PSNR_Gaussian_bila = 22.999 // PSNR_pepper_bila = 13.82

##高斯噪声图片过双边滤波器 denoise 
imag_denoise = re.denoise_bilateral(imag_gaussian, win_size=None, sigma_color=None, sigma_spatial=1, bins=10000, mode='constant', cval=0, multichannel=False)
plt.figure("image CARRE1 apres filtrage de denoise_bilateral gaussian ")
plt.title("image CARRE1 apres filtrage de denoise_bilateral gaussian ")
plt.imshow(imag_denoise,cmap = 'gray')
plt.axis('off')

#椒盐噪声图片过双边滤波器 denoise 
imag_denoise1 = re.denoise_bilateral(imag_pepper, win_size=None, sigma_color=None, sigma_spatial=1, bins=10000, mode='constant', cval=0, multichannel=False)
plt.figure("image CARRE1 apres filtrage de denoise_bilateral pepper ")
plt.title("image CARRE1 apres filtrage de denoise_bilateral pepper ")
plt.imshow(imag_denoise1,cmap = 'gray')
plt.axis('off')

#原图与过了双边滤波器 denoise 的高斯噪声图像/椒盐噪声 信噪比
PSNR_Gaussian_bila1 = skim.compare_psnr(imag_float,imag_denoise,data_range=None, dynamic_range=None)
PSNR_pepper_bila1 = skim.compare_psnr(imag_float,imag_denoise1,data_range=None, dynamic_range=None)
## PSNR_Gaussian_bila1 = 27.27 //  PSNR_pepper_bila1 = 15.07


# 高通滤波器 masque

k = 12
H_PH = np.array([[-1,-2,-1],[-2,k,-2],[-1,-2,-1]],dtype = 'float')
I_ph = sp.ndimage.convolve(imag_float,H_PH)
plt.figure("image CARRE1 apres filtrage PASSE_HAULT k = 15")
plt.title("image CARRE1 apres filtrage PASSE_HAULT k = 15")
plt.imshow(I_ph,cmap = 'gray')
plt.axis('off')



H_PH1 = np.array([[-1,0,-1],[2,0,2],[-1,0,-1]],dtype = 'float')
I_ph1 = sp.ndimage.convolve(imag_float,H_PH1)
plt.figure("image CARRE1 apres filtrage PASSE_HAULT h")
plt.title("image CARRE1 apres filtrage PASSE_HAULT h")
plt.imshow(I_ph1,cmap = 'gray')
plt.axis('off')


H_PH2 = np.array([[-1,2,-1],[0,0,0],[-1,2,-1]],dtype = 'float')
I_ph2 = sp.ndimage.convolve(imag_float,H_PH2)
plt.figure("image CARRE1 apres filtrage PASSE_HAULT v")
plt.title("image CARRE1 apres filtrage PASSE_HAULT v")
plt.imshow(I_ph2,cmap = 'gray')
plt.axis('off')




I_contour = fl.laplace(imag,ksize=3,mask = None)
plt.figure("image CARRE1 apres filtrage laplacien ")
plt.title("image CARRE1 apres filtrage laplacien ")
plt.imshow(I_contour,cmap = 'gray')
plt.axis('off')

I_contourH = fl.sobel_h(imag,mask = None)
plt.figure("image CARRE1 apres filtrage sobel horizontale ")
plt.title("image CARRE1 apres filtrage sobel horizontale ")
plt.imshow(I_contourH,cmap = 'gray')
plt.axis('off')

I_contourV = fl.sobel_v(imag,mask = None)
plt.figure("image CARRE1 apres filtrage sobel verticale ")
plt.title("image CARRE1 apres filtrage sobel verticale ")
plt.imshow(I_contourV,cmap = 'gray')
plt.axis('off')



#
def filtrebutterworthHF(fc,taille,ordre=4):
    s = (taille[0],taille[1])
    H = np.zeros(s)
    [U,V] = np.ogrid[-0.5:0.5:1.0/taille[0],-0.5:0.5:1.0/taille[1]]
    H = 1.0/ (1.0+np.power(fc/np.sqrt(U*U+V*V),2*ordre))
    return H


rows,cols = imag.shape
taille = np.array([rows,cols])


plt.figure("1")
plt.title(" 1 ")
plt.imshow(filtrebutterworthHF(0.25,taille,4),cmap = 'gray')
plt.axis('off')


imag_fft2 = sc.fft2(imag)
imag_fftshift = sc.fftshift(imag_fft2)
#I_ff = imag_fftshift*filtrebutterworthHF(0.25,taille,4)
#I_iff = sc.ifftshift(imag_fftshift)
#I_iff = sc.ifft2(I_iff)
#I_iff = np.abs(I_iff)
plt.figure("2")
plt.title(" 2 ")
plt.imshow(np.abs(imag_fftshift),cmap = 'gray')
plt.axis('off')   
    
#I_iff = sc.ifftshift(I_ff)
#I_iff = sc.ifft2(I_iff)
#plt.figure("1")
#plt.title(" 1 ")
#plt.imshow(I_iff,cmap = 'gray')
#plt.axis('off')



H_bouge = np.array([[0.2,0.2,0.2,0.2,0.2]])
I_bouge = sp.ndimage.convolve(imag_gaussian,H_bouge)
plt.figure("image filtre par filtrage bouge")
plt.title("image filtre par filtrage bouge")
plt.imshow(I_bouge,cmap = 'gray')
plt.axis('off')