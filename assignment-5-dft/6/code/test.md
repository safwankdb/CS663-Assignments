# Assignment 5 DFT: Question 6

We read the given paper *'An FFT-Based Technique for Translation, Rotation, and Scale-Invariant Image Registration'*. Equation 3 from the paper was implemented in Python 3 code to register 2 sets of images as described in the paper.

## Implementation
The following function implements the algorithm using standard NumPy functions.


```python
def register_translation(a, b):
    assert a.shape == b.shape
    h, w = a.shape
    A = F.fft2(a)
    B = F.fft2(b)
    X_Spec =  (A * B.conjugate())/abs(A*B)
    x_spec = np.abs(F.ifft2(X_Spec))
    t_y, t_x = np.unravel_index(np.argmax(x_spec), (h, w))
    t_y = t_y-h if t_y > h//2 else t_y
    t_x = t_x-w if t_x > w//2 else t_x
    return t_x, t_y, x_spec
```

## Case I
The results obtained for the case without any noise were $t_x = -30, t_y = 70$ which is correct.

<div align='center'>
   <img src="Figure_1.png"/>
</div>

## Case II
The results obtained for the case with $ N(0, 20) $ noise were $ t_x = -30, t_y = 70 $ which is correct.

<div align='center'>
   <img src="Figure_2.png"/>
</div>
