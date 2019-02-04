# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 10:16:08 2015

@author: ladretp
"""
   
        
import numpy as np
import skimage as sk
from skimage import color, data,feature
from matplotlib import pyplot as plt
from scipy.ndimage import convolve
import skimage.io as skio

#pour fermer toutes les figures précédentes

plt.close('all')

###############################################################
# filtre 1d de dérivation de Deriche
###############################################################"
def derivation(x,alpha,sx):
    #### les coefficients du filtre
    gain = -(1-np.exp(-alpha))*(1-np.exp(-alpha))/np.exp(-alpha)
    b1_p = gain*np.exp(-alpha)
    a1_p = -2.0*np.exp(-alpha)
    a2_p = np.exp(-2.0*alpha)
    
    b1_n = -gain*np.exp(-alpha)
    a1_n = -2.0*np.exp(-alpha)
    a2_n = np.exp(-2.0*alpha)

    sx = sx-1 #sx = 199
    yd = np.zeros_like(x)
    yd1 = np.zeros_like(x)
    yd2 = np.zeros_like(x)

    yd1[0] = x[0]
    yd1[1] = x[1]
#    yl1[0] = b1_p*x[0]
#    yl1[1] = b0_p*x[1]+b1_p*x[0]
    
#    yl2[sx] = x[sx]
#    yl2[sx-1] = x[sx-1]
    yd2[sx] = x[sx]
    yd2[sx-1] = x[sx-1]
    
    indice = np.arange(2,sx-2,1)
    indice2 = np.arange(sx-2,2,-1)
    
    for i in indice:
        yd1[i] = b1_p*x[i-1]-a1_p*yd1[i-1]-a2_p*yd1[i-2]
    for i in indice2:        
        yd2[i] = b1_n*x[i+1]-a1_n*yd2[i+1]-a2_n*yd2[i+2]
    yd = yd1+yd2

    return yd    

################################################################################    
def lissage(x,alpha,sx):
    # les coefficients du filtre
    gain = (1-np.exp(-alpha))*(1-np.exp(-alpha))/(1+2.0*np.exp(-alpha)-np.exp(-2.0*alpha))
    b0_p = gain;
    b1_p = gain*(alpha-1)*np.exp(-alpha)
    a1_p = -2.0*np.exp(-alpha)
    a2_p = np.exp(-2.0*alpha);
    
    b1_n = gain*(alpha+1)*np.exp(-alpha)
    b2_n = -gain*np.exp(-2.0*alpha)
    a1_n = -2.0*np.exp(-alpha)
    a2_n = np.exp(-2.0*alpha)
    
    sx = sx-1 #sx = 199
    yl = np.zeros_like(x)
    yl1 = np.zeros_like(x)
    yl2 = np.zeros_like(x)
    
#    yl1[0] = x[0]
#    yl1[1] = x[1]
    yl1[0] = b0_p*x[0]
    yl1[1] = b0_p*x[1]+b1_p*x[0]
    
    yl2[sx] = x[sx]
    yl2[sx-1] = x[sx-1]
#    yl2[sx] = x[sx]
#    yl2[sx-1] = x[sx-1]
    
    indice = np.arange(2,sx-2,1)
    indice2 = np.arange(sx-2,2,-1)
    for i in indice:
        yl1[i] = b0_p*x[i]+b1_p*x[i-1]-a1_p*yl1[i-1]-a2_p*yl1[i-2]
    for i in indice2:        
        yl2[i] = b1_n*x[i+1]+b2_n*x[i+2]-a1_n*yl2[i+1]-a2_n*yl2[i+2]
    yl = yl1+yl2
    
    return yl    

###################################################################################"

# Debut programme principal

#######################################################################

#creation image test simple
#lena=np.ones((256,256))
#lena[100:200,:]=0.0

#differents choix pour l'image
lena=data.coins()
lena=data.checkerboard()
#lena = skio.imread('D:\image\cameraman.tif')
plt.imshow(lena,cmap='gray')
lena=color.rgb2gray(lena)

lena=sk.img_as_float(lena)  #le signal est normalisé entre 0 et 1
tailleinit=np.shape(lena)


#######################################################################"
#Creation d'un Bruit gaussien 2D correspond à N(m,sigma^2)
m=0.0
sigma=0.9
noise = sigma*np.random.randn(tailleinit[0],tailleinit[1])+m

plt.figure("noise ")
plt.title("noise ")
plt.imshow(noise,cmap = 'gray')
plt.axis('off')

im_br=lena+noise
im_br=sk.img_as_float(im_br)
plt.figure("im_br"), plt.imshow(im_br,cmap='gray'),plt.title("im_br ")

##########################################################################
#ce sera l'image traitée, pour supprimer le bruit alors mettre en #
#commentaire la ligne suivante :
#lena=im_br

taille=np.shape(lena)
taillex=taille[0]
tailley=taille[1]

###############################################################
# Implémenter Canny version Deriche  contours Ix#
# les paramètres de l'algo phase de lissage
###############################################################
alpha=1  # paramètre à modifier pour l'étude du filtre
##################################################################"



y=np.zeros((taillex,tailley))

J= np.arange(0,tailley,1) #j = 0,1,2,3...199
    
for i in J:
    y[:,i]=lissage(lena[:,i],alpha,taillex)

plt.figure(80)
plt.imshow(y,cmap='gray')
plt.title("lissage Deriche")

plt.figure(90)
plt.plot(y[:,105])
plt.plot(lena[:,105],'g-')
plt.title("lissage Deriche")
#
##########################################################################
## phase de derivation
########################################################################
#
#
yde=np.zeros((taillex,tailley))
I=np.arange(0,taillex,1)

for i in I:
    x=y[i,:]
    yde[i,:]=derivation(x,alpha,tailley)

Ix=yde
#
################################################################
## Lissage dans l'autre sens
################################################################
#
y=np.zeros((taillex,tailley))

I=np.arange(0,taillex,1)
J = np.arange(0, tailley, 1)

for i in J:
    y[i,:]=lissage(lena[:,i],alpha,taillex)

plt.figure(81)
plt.imshow(y,cmap='gray')
plt.title("lissage Deriche")



plt.figure(91)
plt.plot(y[105,:])
plt.plot(lena[105,:],'g-')
plt.title("lissage Deriche")

##########################################################################
## phase de derivation
########################################################################
#
yde=np.zeros((taillex,tailley))


for i in J:
    x=y[:,i]
    yde[:,i]=derivation(x,alpha,taillex)

plt.figure(102)
plt.imshow(yde,cmap='gray')
plt.title("derivation Deriche ")
#
plt.figure(111)
plt.plot(yde[:,105],'r+')
plt.plot(y[:,105])   

Iy=yde
##################################################################################"""




Norme=Ix*Ix+Iy*Iy
Norme_Visu=sk.img_as_float(Norme) 

if tailleinit[0]<taillex :
    Norme2=Norme[tailleinit[0]:taillex-tailleinit[0],tailleinit[1]:tailley-tailleinit[1]]
else :
    Norme2=Norme

plt.figure(200), plt.imshow(Norme2,cmap='gray')
plt.title('Norme Deriche')
th_Normbin = sk.filters.threshold_otsu(Norme2)

Normbin = sk.morphology.skeletonize(Norme2 >= th_Normbin)


plt.figure(211), plt.imshow(Normbin,cmap='gray')
plt.title('contour binaire Deriche')

