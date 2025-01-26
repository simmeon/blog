'''
Plotting the correlation function of the digital signal stationary random process example.
Created by: simmeon
Last Modified: 2025-01-25
License: MIT

'''

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')

# Array of tau values to calculate the correlation function over
tau = np.linspace(-5, 5, 500)

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

plt.plot(tau, R)

plt.title('Correlation Function of Digital Signal', fontsize=20)
plt.xlabel(r'$\tau$', fontsize=20)
plt.ylabel(r'$R_{xx}(\tau)$', fontsize=20)
plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
plt.xticks(np.arange(-5, 6, 1))
plt.ylim((0, 1))

plt.show()