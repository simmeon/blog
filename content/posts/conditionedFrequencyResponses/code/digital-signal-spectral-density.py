'''
Plotting the spectral density function of the digital signal stationary random process example.
Created by: simmeon
Last Modified: 2025-01-25
License: MIT

'''

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import zoom_fft

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')

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
G = abs(G) / n

# Frequencies corresponding to Fourier transform data
f = np.linspace(0, 1, len(R)//2)

# zoom_fft gives only positive freqs, aka. one-sided
# since Fourier is symmetric, we can flip it for negative freqs
S_negative = np.flip(G)
f_negative = - np.flip(f)

G = G.tolist()
S_negative = S_negative.tolist()
f = f.tolist()
f_negative = f_negative.tolist()

# Adding positive and negative freqs and hlaving amplitude gives full two-sided spectral density
S = S_negative + G
f = f_negative + f
S = np.array(S)
S = S / 2

print(G)

# Plotting
fig = plt.figure()

ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

ax1.plot(f[n//2:], S[n//2:], color='white')
ax1.plot(f[0:n//2], S[0:n//2], color='white')

ax2.plot(f[n//2:], G, color='white')

ax1.set_title('Two and One-sided Spectral Density Functions of Digital Signal', fontsize=20)
ax2.set_xlabel('f', fontsize=20)
ax1.set_ylabel(r'$S_{xx}(f)$', fontsize=20)
ax2.set_ylabel(r'$G_{xx}(f)$', fontsize=20)
ax1.set_xlim((-1, 1))
ax2.set_xlim((-1, 1))
ax1.set_ylim((0, 0.3))
ax2.set_ylim((0, 0.3))
ax1.set_xticks(np.arange(-1, 1.1, 0.1))
ax2.set_xticks(np.arange(-1, 1.1, 0.1))

plt.show()