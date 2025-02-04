'''
A simple two-input (partically correlated), one-output linear system to introduce 
the problems that arise with partially correlated inputs.

Last Modified: 2025-02-04
License: MIT

'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, zoom_fft
from scipy.integrate import odeint

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')


pi = np.pi

# ------------------------------------------------------------------------------

# Define and simulate system

# System coefficienets (this is secret don't look...)
a = 2
b = 3
c = 2
d = 0.5
k = (2 * pi * 2)**2 # ~ 158

# Simulation properties
T = 50
fs = 1000
N = T * fs
t = np.linspace(0, T, N, endpoint=False)

# Define input x1: freq sweep
fmin = 0.1
fmax = 5
x1 = chirp(t, fmin, T, fmax)

# Define input x2: partially correlated
x2_uc = np.random.normal(0, 1, N)
x2 = d * x1 + x2_uc

# Define state derivative
def dydt(y, t):
    u1 = x1[int(t * fs) - 1]
    u2 = x2[int(t * fs) - 1]
    return np.array([y[1], - k * y[0] - c * y[1] + a * u1 + b * u2])
    
# Simulate system
y0 = [0, 0]
y = odeint(dydt, y0, t)
y = y[:,0]

# ------------------------------------------------------------------------------

# Frequency Responses

# Analytical frequency response
f = np.logspace(np.log10(fmin), np.log10(fmax), N)
s = 1.0j * 2 * pi * f

H1y = a / (s**2 + c * s + k)
H2y = b / (s**2 + c * s + k)

# Estimated response
Y = zoom_fft(y, [fmin, fmax], m=N, fs=fs)
X1 = zoom_fft(x1, [fmin, fmax], m=N, fs=fs)
X2 = zoom_fft(x2, [fmin, fmax], m=N, fs=fs)

H1y_est = Y / X1
H2y_est = Y / X2

f_est = np.linspace(fmin, fmax, N)

# ------------------------------------------------------------------------------

# Plotting
fig = plt.figure()

ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,3)
ax3 = fig.add_subplot(2,2,2)
ax4 = fig.add_subplot(2,2,4)


ax1.set_title(r'$Y / X_{1}$ Magnitude')
ax1.semilogx(f, 20 * np.log10(abs(H1y)), label='Analytical')
ax1.semilogx(f_est, 20 * np.log10(abs(H1y_est)), label='Estimated')

ax2.set_title(r'$Y / X_{1}$ Phase')
ax2.semilogx(f, np.rad2deg(np.angle(H1y)), label='Analytical')
ax2.semilogx(f_est, np.rad2deg(np.angle(H1y_est)), label='Estimated')

ax3.set_title(r'$Y / X_{2}$ Magnitude')
ax3.semilogx(f, 20 * np.log10(abs(H2y)), label='Analytical')
ax3.semilogx(f_est, 20 * np.log10(abs(H2y_est)), label='Estimated')

ax4.set_title(r'$Y / X_{2}$ Phase')
ax4.semilogx(f, np.rad2deg(np.angle(H2y)), label='Analytical')
ax4.semilogx(f_est, np.rad2deg(np.angle(H2y_est)), label='Estimated')

ax2.set_xlabel('Frequency [Hz]')
ax4.set_xlabel('Frequency [Hz]')

ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()

plt.show()