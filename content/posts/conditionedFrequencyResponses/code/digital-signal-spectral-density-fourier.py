'''
Estimated spectral density function of the digital signal from direct finite Fourier transforms.
Created by: simmeon
Last Modified: 2025-01-26
License: MIT

'''

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import zoom_fft

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')

# Same proper G from last example
# Array of tau values to calculate the correlation function over
n = 5000
T = 5

tau = np.linspace(-T/2, T/2, n)

# Correlation function
# NOTE: this is not calculated properly in any way,
# it is quick and dirty but gives the correct result
def Rxx(tau_array):
    Rxx = []
    for tau in tau_array:
        if abs(tau) < 1:
            Rxx.append(((1 - abs(tau)) / 1) * 0.25 + 0.25)
        else:
            Rxx.append(0.25)
    
    return Rxx

R = Rxx(tau)

# Fourier transform of correlation function using a fancy chirp-z tranform method
# for better frequency resolution
G = zoom_fft(R, [0, 1], m=len(R)//2, fs=n/(2*T))
G_actual = abs(G) / len(R)
f_actual = np.linspace(0, 1, len(R)//2)

# --------------------------------------------------------------------- #
G_array = []
for i in range(100):
    num_bits = 10

    # Calculate random series of bits
    bits = np.random.randint(0, 2, num_bits)

    N = 100 # Number of sample points per bit
    bit_length = 1 # seconds

    # Make length of each bit 1 s
    T = num_bits * bit_length
    t = np.linspace(0, T, num_bits * N)

    # sample function
    x = []

    # Make sample function have N points per bit and add offsets
    for bit in bits:
        x = x + [bit] * N

    # Fourier transform sample function
    X = zoom_fft(x, [0, 1], m=len(x)//2, fs=N)
    X = abs(X) / len(x) * 2
 
    # Calculate estimated spectral density
    G = 2 / T * (np.conjugate(X) * X)
    G_array.append(G)


# Frequencies corresponding to Fourier transform data
f = np.linspace(0, 1, len(x)//2)

for G in G_array:
    plt.plot(f, G, alpha=0.05)

plt.plot(f_actual, G_actual, label='Actual')

plt.title('100 Estimates of Spectral Density Function', fontsize=20)
plt.xlim((0, 1))
plt.xticks(np.arange(0, 1.1, 0.1))
plt.ylabel(r'$G_{xx}(f)$', fontsize=20)
plt.xlabel('f', fontsize=20)
plt.legend(fontsize=20)

plt.show()