import numpy as np
import numpy.fft as F
from matplotlib import pyplot as plt
import cv2


def register_translation(a, b):
    assert a.shape == b.shape
    h, w = a.shape
    A = F.fft2(a)
    B = F.fft2(b)
    X_Spec = (A * B.conjugate())/abs(A*B)
    x_spec = np.abs(F.ifft2(X_Spec))
    t_y, t_x = np.unravel_index(np.argmax(x_spec), (h, w))
    t_y = t_y-h if t_y > h//2 else t_y
    t_x = t_x-w if t_x > w//2 else t_x
    return t_x, t_y, x_spec


def show_results(a, b, x_spec):
    plt.figure(figsize=(10, 10))
    plt.subplot(221)
    plt.title('Image 1')
    plt.imshow(a, cmap='gray')
    plt.subplot(222)
    plt.title('Image 2')
    plt.imshow(b, cmap='gray')
    plt.subplot(223)
    plt.imshow(x_spec, cmap='gray')
    plt.title('Magnitude of cross spectrum')
    plt.subplot(224)
    plt.imshow(np.log(x_spec), cmap='gray')
    plt.title('Log magnitude of cross spectrum')
    plt.tight_layout()
    plt.show()


a = cv2.imread('../images/img1.jpg', 0)
b = cv2.imread('../images/img2.jpg', 0)

t_x, t_y, x_spec = register_translation(a, b)
show_results(a, b, x_spec)
print('The translation required to register images is : {}'.format((t_x, t_y)))

a = a + 20*np.random.randn(*a.shape)
b = b + 20*np.random.randn(*b.shape)
t_x, t_y, x_spec = register_translation(a, b)
show_results(a, b, x_spec)
print('The translation required to register noisy images is : {}'.format((t_x, t_y)))
