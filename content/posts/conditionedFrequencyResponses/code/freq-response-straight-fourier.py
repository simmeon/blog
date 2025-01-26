'''
Seeing how noisy data can affect direct calculation of the frequency
response from Fourier transforms.

Created by: simmeon
Last Modified: 2025-01-25
License: MIT

'''

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.fft import rfft, rfftfreq


plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')

pi = np.pi

# Let's set up a simple spring, mass, damper system to mess around with.
fn = 1              # natural frequency of system [Hz]
zeta = 0.05          # system damping ratio

wn = 2 * pi * fn    # natural freq [rad/s]

# We can find the analytical transfer function to get the true frequency response
def H_func(f):
    s = 1.0j * 2 * pi * f
    return wn**2 / (s**2 + 2 * zeta * wn * s + wn**2)

fmin = 0.1  # min frequency in response
fmax = 10   # max frequency in response
f_analytical = np.logspace(np.log10(fmin), np.log10(fmax), 500) # array of freqs to calculate response for

H_analytical = H_func(f_analytical)                     # Frequency response
mag_analytical = 20 * np.log10(abs(H_analytical))       # Magnitude of response (in dB)
phase_analytical = np.rad2deg(np.angle(H_analytical))   # Phase of response (in deg)


# Now let's create some noisy input and output data to simulate measurements
# To do this we will solve the system with a frequency sweep over the range of interest (0.1 - 10 Hz)

T = 20  # period to simulate system over [s]

# Define our input function, a frequency sweep
# NOTE: to get a correct frequency response over all frequencies of interest, the input must
# necessarily contaion information on all the frequencies you are interested in. 
def u_func(t):
    f = t / T * fmax * 2
    return 10 * np.sin(2 * pi * f * t)

# Define the derivative function of the system (state space derivative)
def dxdt(x, t):
    u = u_func(t)
    return np.array([x[1], - wn**2 * x[0] - 2 * zeta * wn * x[1] + wn**2 * u])


# Solve system
N = 5000                   # number of sample points
t = np.linspace(0, T, N)    # time array
x0 = np.array([0, 0])       # initial state conditions
sol = odeint(dxdt, x0, t)

y = sol[:, 0]   # pull out position from state as our output
x = u_func(t)   # get array of inputs for use later

# Now we will add some output noise to simulate noisy measurements
y = y + np.random.normal(0, 5, len(y))

# Then, we can fourier transform our input and output to calculate the freqency response
Y = rfft(y)
X = rfft(x)
f_estimated = rfftfreq(N, T / N)
H_estimated = Y / X                                   # Estimated freq response from noisy data
mag_estimated = 20 * np.log10(abs(H_estimated))       # Magnitude of response (in dB)
phase_estimated = np.rad2deg(np.angle(H_estimated))   # Phase of response (in deg)


# Plotting
plt.figure(figsize=(12, 8))

# Subplot 1: Time domain response
plt.subplot(3, 1, 1)
plt.plot(t, y)
plt.title("System Response (y) over Time", fontsize=20)
plt.xlabel("Time (s)", fontsize=18)
plt.ylabel("Output (y)", fontsize=18)
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=16)

# Subplot 2: Bode magnitude plot
plt.subplot(3, 1, 2)
plt.semilogx(f_estimated, mag_estimated, label="Estimated", linewidth=2)
plt.semilogx(f_analytical, mag_analytical, label="Analytical", linestyle='--', linewidth=2)
plt.title("Bode Magnitude Plot", fontsize=20)
plt.ylabel("Magnitude (dB)", fontsize=18)
plt.legend(fontsize=16)
plt.grid(True, which='both', axis='both')
plt.xscale('log')
plt.xlim((fmin, fmax))
plt.tick_params(axis='both', which='major', labelsize=12)

# Subplot 3: Bode phase plot
plt.subplot(3, 1, 3)
plt.semilogx(f_estimated, phase_estimated, label="Estimated", linewidth=2)
plt.semilogx(f_analytical, phase_analytical, label="Analytical", linestyle='--', linewidth=2)
plt.title("Bode Phase Plot", fontsize=20)
plt.ylabel("Phase (Degrees)", fontsize=18)
plt.xlabel("Frequency (Hz)", fontsize=18)
plt.legend(fontsize=16)
plt.grid(True, which='both', axis='both')
plt.xscale('log')
plt.xlim((fmin, fmax))
plt.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()
plt.show()
