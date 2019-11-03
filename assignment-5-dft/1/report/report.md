# Assignment 5 DFT : Question 1

## Team Members
- Mohd Safwan
- Manas Vasishtha
- Sumrit Gupta

## Solution

We know that,

$$g_1 = f_1 + h_2 * f_2, g_2 = h_1*f_1 + f_2$$

We will now analyze these equations in the frequency domain (Fourier Transform).

$$G_1 = F_1+H_2F_2, G_2 = H_1F_1+F_2$$

This is a pair of linear equations in 2 variables. We solve for $F_1$ and $F_2$ to get

$$ F_1 = \frac{G_1-H_2G_2}{1-H_1H_2} , F_2 = \frac{G_2-H_1G_1}{1-H_1H_2} $$

Now the answer is simply
$$ f_1 = F^{-1}(F_1),  f_2 = F^{-1}(F_2) $$

## Inherent Problem in Solution

We know that $h_1$ and $h_2$ are blurring kernels. Their fourier transforms $H_1$ and $H_2$ act as low pass filters and do not tend to amplify images. Thus, for low frequencies, the term $H_1H_2$ approaches $1$, so the denomiator approaches $0$. This is not desirable because this will amplify noise for low frequencies.
