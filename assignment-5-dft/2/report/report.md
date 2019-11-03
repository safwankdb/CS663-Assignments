# Assignment 5 DFT: Question 2

## Team Members
- Mohd Safwan
- Manas Vasishtha
- Sumrit Gupta

# Solution

Since we have $g=h*f$, in the frequency domain, we get $G=HF$. So, the trivial solution is
$$ f = F^{-1}\bigg(\frac{G}{H}\bigg) $$

We know that gradient operators act as high pass filters and so $H \rightarrow 0$ for low frequencies. Recovering high frequency components will not be difficult. But, in the presence of noise, the small denominator will amplify noise which will make recovering those components difficult. However, even if noise is not present, there is another problem which we will try to analyze the problems with examples.

## Problems
### 1D Case
Let us assume a 1D image of length $L$. Also, we assume the zero crossing kernel as our gradient kernel of length $3$.

\begin{align*}
h &= \begin{bmatrix} -1 & 0 & 1\end{bmatrix}\\
H(n) &= \sum_{k} h[k]e^{-j\frac{2\pi}{L}nk}\\
&= -1 + e^{-j\frac{4\pi}{L}n}, n \in \{0, 1, 2~...~L-1\}\\
\end{align*}

We now have $H(0)=0$ which means there is no DC component after the gradient operation. So we can not recover the average grayscale level of the image.

### 2D Case
Similar to the 1D case, we attempt to calculate the DFT of the 2-D kernel. We use the Sobel kernels 

$$
h_x = \begin{bmatrix}-1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}
,
h_y = \begin{bmatrix}1 & 2 & 1 \\ 0 & 0 & 0 \\ -1 & -2 & -1 \end{bmatrix}
$$

Assuming we need to take a $L \times L$ DFT after zero padding we obtain,

\begin{align*}
H_x(n_1, n_2) &= \sum_{x} \sum_{y} h_x[x,y]e^{-j\frac{2\pi}{L}n_1x}e^{-j\frac{2\pi}{L}n_2y}\\
&= (-1 + e^{-j\frac{4\pi}{L}n_1})(1 + 2e^{-j\frac{2\pi}{L}n_2} + e^{-j\frac{4\pi}{L}n_2})\\
\\
H_y(n_1, n_2) &= \sum_{x} \sum_{y} h_y[x,y]e^{-j\frac{2\pi}{L}n_1x}e^{-j\frac{2\pi}{L}n_2y}\\
&= (1 - e^{-j\frac{4\pi}{L}n_2})(1 + 2e^{-j\frac{2\pi}{L}n_1} + e^{-j\frac{4\pi}{L}n_1})\\
\\
&n_1, n_2 \in \{0, 1, 2~...~L-1\}\\
\end{align*}

Quite clearly, for the DC case we obtain $H_x(0,0) = H_y(0,0) = 0$ and we won't be able to recover the DC components correctly. Image can't be recovered completely without knowing the average grayscale level.