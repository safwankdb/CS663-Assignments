#!/usr/bin/env python
# coding: utf-8
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import cv2


def gaussian_mask(n, sigma=None):
    if sigma is None:
        sigma = 0.3*(n//2) + 0.8
    X = np.mgrid[:n]
    Y, X = np.mgrid[:n, :n]
    Y = Y - n//2
    X = X - n//2
    kernel = np.exp(-(X**2+Y**2)/(2*sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel


def image_gradient(I):
    h, w = I.shape
    sobel_x = np.array([[-1,  0,  1],
                        [-2,  0,  2],
                        [-1,  0,  1]])
    sobel_y = np.array([[1,  2,  1],
                        [0,  0,  0],
                        [-1, -2, -1]])
    I_x = np.zeros(I.shape)
    I_y = np.zeros(I.shape)
    for y in range(1, h-1):
        for x in range(1, w-1):
            patch = I[y-1:y+2, x-1:x+2]
            I_x[y, x] = np.sum(patch * sobel_x)
            I_y[y, x] = np.sum(patch * sobel_y)
    return I_x, I_y


def solve(I_x, I_y, n, sigma):
    kernel = gaussian_mask(n, sigma)
    L_1 = np.zeros(I_x.shape)
    L_2 = np.zeros(I_x.shape)
    h, w = I_x.shape
    print('Calculating Structure Tensor and Eigenvalues')
    for y in range(n, h-n):
        for x in range(n, w-n):
            A = np.zeros((2, 2))
            for i in range(-n//2, n//2+1):
                for j in range(-n//2, n//2+1):
                    W = kernel[i, j]
                    D = np.array([I_x[y+i, x+j], I_y[y+i, x+j]])
                    D = D.reshape(2, 1)
                    D = D.dot(D.T)
                    A += W * D
            l1, l2 = np.linalg.eigvals(A)
            L_1[y, x] = l1
            L_2[y, x] = l2
        print('Progress: {:.02f}{}'.format(100*(y+1-n)/(h-2*n), '%'), end='\r')
    return L_1, L_2


def get_cornerness(L_1, L_2, k):
    C = L_1*L_2 - k*(L_1+L_2)**2
    return C


img = Image.open('data/boat.jpg')
img = np.array(img)
img = (img - img.min())/(img.max()-img.min())
I_x, I_y = image_gradient(img)
L_1, L_2 = solve(I_x, I_y, 5, 2)
C = get_cornerness(L_1, L_2, 0.05)


plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(img, cmap='gray')
plt.subplot(132)
plt.imshow(I_x, cmap='gray')
plt.subplot(133)
plt.imshow(I_y, cmap='gray')
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 10))
plt.subplot(121)
plt.imshow(L_1, cmap='gray')
plt.subplot(122)
plt.imshow(L_2, cmap='gray')
plt.show()


filename = 'data/boat.jpg'
C = get_cornerness(L_1, L_2, 0.025)
plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.imshow(C)
plt.title('Our Implementation')
plt.colorbar()
plt.subplot(122)
gray = cv2.imread(filename, 0)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 5, 3, 0.025)
plt.imshow(dst)
plt.title('OpenCV Implementation')
plt.colorbar()
plt.show()


plt.imshow(gaussian_mask(5))
plt.show()
