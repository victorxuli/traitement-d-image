# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 17:01:40 2018

@author: victo
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from skimage.feature import canny
from skimage import data

import matplotlib.pyplot as plt
from matplotlib import cm
from skimage import filters as fl
import skimage.io as skio
import skimage
import skimage.morphology as sm
plt.close("all")
# Constructing test image
#image = np.zeros((100, 100))
#idx = np.arange(25, 75)
#image[idx[::-1], idx] = 255
#image[idx, idx] = 255
image = skio.imread(r'D:\image\chemin-de-fer.jpg')
image = skimage.color.rgb2gray(image)
#image = data.checkerboard()
# Classic straight-line Hough transform
h, theta, d = hough_line(image)

# Generating figure 1
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
ax = axes.ravel()

ax[0].imshow(image, cmap=cm.gray)
ax[0].set_title('Input image')
ax[0].set_axis_off()

ax[1].imshow(np.log(1 + h),
             extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
             cmap=cm.gray, aspect=1/1.5)
ax[1].set_title('Hough transform')
ax[1].set_xlabel('Angles (degrees)')
ax[1].set_ylabel('Distance (pixels)')
ax[1].axis('image')

ax[2].imshow(image, cmap=cm.gray)
for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
    y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
    y1 = (dist - image.shape[1] * np.cos(angle)) / np.sin(angle)
    ax[2].plot((0, image.shape[1]), (y0, y1), '-r')
ax[2].set_xlim((0, image.shape[1]))
ax[2].set_ylim((image.shape[0], 0))
ax[2].set_axis_off()
ax[2].set_title('Detected lines')

plt.tight_layout()
plt.show()

# Line finding using the Probabilistic Hough Transform
#image = data.camera()
image = skio.imread(r'D:\image\chemin-de-fer.jpg')
image = skimage.color.rgb2gray(image)
#I=sm.black_tophat(image,sm.square(40))
#plt.figure(2)
#plt.imshow(I,cmap= 'gray')
#image = data.checkerboard()
edges = fl.scharr_v(image)
for i in range(edges.shape[0]):
    for j in range(edges.shape[1]):
        if(np.abs(edges[i,j])<0.54):
            edges[i,j] = 0
        else:
            edges[i,j] = 1
#edges = fl.roberts(image)
#plt.figure("image top hat ")
#plt.title("image top hat ")
#plt.imshow(edges,cmap = 'gray')
#plt.axis('off')
lines = probabilistic_hough_line(edges, threshold=80, line_length=120,
                                 line_gap=12)

# Generating figure 2
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap=cm.gray)
ax[0].set_title('Input image')

ax[1].imshow(edges, cmap=cm.gray)
ax[1].set_title('Canny edges')

ax[2].imshow(edges * 0)
for line in lines:
    p0, p1 = line
    ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
ax[2].set_xlim((0, image.shape[1]))
ax[2].set_ylim((image.shape[0], 0))
ax[2].set_title('Probabilistic Hough')

for a in ax:
    a.set_axis_off()

plt.tight_layout()
plt.show()