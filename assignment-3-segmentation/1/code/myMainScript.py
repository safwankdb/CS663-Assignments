from numpy import array
from matplotlib import pyplot as plt
from PIL import Image

from myHarrisCornerDetector import corner_harris

img_path = '../data/test.jpg'
img = array(Image.open(img_path))
img = (img - img.min())/(img.max()-img.min())
C, I_x, I_y, L_1, L_2 = corner_harris(img, k=0.06)
C = (C - C.min())/(C.max()-C.min())

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.title('$I_x$')
plt.imshow(I_x, cmap='gray')
plt.subplot(122)
plt.title('$I_y$')
plt.imshow(I_y, cmap='gray')
plt.tight_layout()
plt.show()


plt.figure(figsize=(16, 5))
plt.subplot(121)
plt.title(r'$\lambda_1$')
plt.imshow(L_1, cmap='gnuplot')
plt.colorbar()
plt.subplot(122)
plt.title(r'$\lambda_2$')
plt.imshow(L_2, cmap='gnuplot')
plt.colorbar()
plt.tight_layout()
plt.show()


plt.figure(figsize=(13, 5))
plt.subplot(121)
plt.imshow(C-0.457, cmap='gnuplot')
plt.title('Corner-ness Map')
plt.colorbar()
plt.subplot(122)
plt.imshow(img/2+2*C*(C >= 0.457), cmap='gnuplot')
plt.title('Detected Corners')
plt.tight_layout()
plt.show()
