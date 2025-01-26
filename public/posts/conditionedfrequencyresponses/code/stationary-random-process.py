'''
Exploring what a stationary random process is and some useful features of it.
Created by: simmeon
Last Modified: 2025-01-25
License: MIT

'''

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import histogram

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')

# We will want to store each 'sample function' in an array
x = []

N = 50 # sample length
T = 10 # time period sample is over
t = np.linspace(0, T, N)
num_samples = 100 # number of sample functions to run

for i in range(num_samples):
    # The stationary random process we will use is random normal noise
    x.append(np.random.normal(0, 1, N))


# We can then estimate the probability density function at a particular time (t)
# by counting how many of the sample functions have values within a particular 
# range at that time.
ti_data = np.zeros((N, num_samples))
pdfs = []
num_bins = 20
pdf_vals = np.linspace(-5, 5, num_bins)
for i in range(N):
    for j in range(num_samples):
        ti_data[i, j] = (x[j][i])
    pdfs.append(histogram(ti_data[i], -5, 5, num_bins) / num_samples)


# Plotting
fig= plt.figure()

ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 3)
ax3 = fig.add_subplot(2, 2, 2)
ax4 = fig.add_subplot(2, 2, 4)

ax1.set_title("One Sample Function", fontsize=20)
ax1.set_xlabel("t", fontsize=20)
ax1.set_ylabel(r"$x_{1}(t)$", fontsize=20)
for i in range(1):
    ax1.plot(t, x[i], alpha=1.0)

ax1.set_ylim((-4, 4))


ax2.set_title(r"Probability Density Function for $t = 0$", fontsize=20)
ax2.set_xlabel(r"$x_{k}(t = 0$)", fontsize=20)
ax2.set_ylabel(r"$f_{X}(x)$", fontsize=20)
for i in range(1):
    ax2.plot(pdf_vals, pdfs[i], alpha=1.0)

ax2.set_ylim((0, 0.3))


ax3.set_title(f"{num_samples} Sample Functions", fontsize=20)
ax3.set_xlabel("t", fontsize=20)
ax3.set_ylabel(r"$x_{k}(t)$", fontsize=20)
for i in range(num_samples):
    ax3.plot(t, x[i], alpha=0.05)

ax3.plot(t, x[0], color='yellow')

ax3.set_ylim((-4, 4))


ax4.set_title(r"Probability Density Functions for each $t_{i}$", fontsize=20)
ax4.set_xlabel(r"$x(t_{i}$)", fontsize=20)
ax4.set_ylabel(r"$f_{X}(x)$", fontsize=20)
for i in range(N):
    ax4.plot(pdf_vals, pdfs[i], alpha=0.1)

ax4.plot(pdf_vals, pdfs[0], alpha=1, color='yellow')

ax4.set_ylim((0, 0.3))

plt.show()