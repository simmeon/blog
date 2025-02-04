"""
A simple two-input (partically correlated), one-output linear system to test 
with.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp
from scipy.integrate import odeint
from scipy.optimize import minimize

import frequency_response

pi = np.pi

# ------------------------------------------------------------------------------

# Define and simulate system

# Output noise standard deviation
output_noise_sd = 0.01

# System coefficienets
a = 2
b = 3
c = 2
d = 0.2
k = (2 * pi * 2)**2 # ~ 158

# Simulation properties
T = 100
fs = 1000
N = T * fs
t = np.linspace(0, T, N, endpoint=False)

# Define input x1: freq sweep
fmin_in = 0
fmax_in = 5
x1 = chirp(t, fmin_in, T, fmax_in)

# Define input x2: no particular reason I made it be this, just trying to roughly cover freq range
# x2_uc = np.sin(2*pi*0.2*t) + np.sin(2*pi*0.37*t) + np.sin(2*pi*1.2*t) + np.sin(2*pi*1.7*t) + np.sin(2*pi*2.0*t) + np.sin(2*pi*2.6*t) + np.sin(2*pi*3.0*t) + np.sin(2*pi*3.3*t) + np.sin(2*pi*4.2*t)
x2_uc = np.random.normal(0, 1, len(t))
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

# Add noise to output
y = y + np.random.normal(0, output_noise_sd, len(y))


# ------------------------------------------------------------------------------

# Frequency Responses
fmin = 0.1
fmax = 5

# Analytical frequency response
f = np.logspace(np.log10(fmin), np.log10(fmax), N)
s = 1.0j * 2 * pi * f

H1y = a / (s**2 + c * s + k)
H2y = b / (s**2 + c * s + k)


# SISO frequency response, single window size
Twindow = 20
overlap = 0.8

x1_windows = frequency_response.create_hann_windows(x1, overlap, Twindow, fs)
x2_windows = frequency_response.create_hann_windows(x2, overlap, Twindow, fs)
y_windows = frequency_response.create_hann_windows(y, overlap, Twindow, fs)

points, G11, Gyy, G1y = frequency_response.avg_spectral_estimates(x1_windows, y_windows, Twindow, fmin, fmax)
points, G22, Gyy, G2y = frequency_response.avg_spectral_estimates(x2_windows, y_windows, Twindow, fmin, fmax)
points, G11, G22, G12 = frequency_response.avg_spectral_estimates(x1_windows, x2_windows, Twindow, fmin, fmax)
G21 = np.conjugate(G12)

f_est = np.angle(points)*fs/(2*pi)

H1y_siso_est = G1y / G11
H2y_siso_est = G2y / G22

C1y = abs(G1y)**2 / (G11 * Gyy)
C2y = abs(G2y)**2 / (G22 * Gyy)

# MISO freq response
Gxx_matrix = np.array([
    [G11, G12], 
    [G21, G22]
])

Gxy_matrix = np.array([
    [G1y], 
    [G2y]
])

Hxy_matrix = []
for f_idx in range(len(points)):
    Hxy_matrix.append(np.linalg.solve(Gxx_matrix[:, :, f_idx], Gxy_matrix[:, :, f_idx]))

Hxy_matrix = np.array(Hxy_matrix)

H1y_miso_est = Hxy_matrix[:, 0]
H2y_miso_est = Hxy_matrix[:, 1]


# Conditioned spectral quantities

# From Otnes and Enochson formulation
Gy1 = np.conjugate(G1y)
Gy2 = np.conjugate(G2y)

# For input x1
Gyxx = np.array([
    [Gyy, Gy1, Gy2],
    [G1y, G11, G12], 
    [G2y, G21, G22]
])

Gyxx = np.swapaxes(Gyxx, 0, 2)
Gyxx = np.swapaxes(Gyxx, 1, 2)

Zyy = Gyxx[:, 0:2, 0:2]
Zy1 = Gyxx[:, 0:2, 2:]
Z1y = Gyxx[:, 2:, 0:2]
Z11 = Gyxx[:, 2:, 2:]

# Conditioned x for other p inputs
Gxy_p = Zyy - np.matmul(Zy1, np.matmul(np.linalg.inv(Z11), Z1y))

Gyy_p = Gxy_p[:, 0, 0]
G11_p = Gxy_p[:, 1, 1]
G1y_p = Gxy_p[:, 1, 0]

H1y_miso_2 = G1y_p / G11_p
C1y_p = abs(G1y_p)**2 / (G11_p * Gyy_p)

# For input x2
# Swap column 2 with column (2+1)
# Swap row 2 with row (2+1)
Gyxx[:, :, [1, 2]] = Gyxx[:, :, [2, 1]]
Gyxx[:, [1, 2], :] = Gyxx[:, [2, 1], :]
# In general for input xk,
# Swap column 2 with column (k+1)
# Swap row 2 with row (k+1)

Zyy = Gyxx[:, 0:2, 0:2]
Zy1 = Gyxx[:, 0:2, 2:]
Z1y = Gyxx[:, 2:, 0:2]
Z11 = Gyxx[:, 2:, 2:]

# Conditioned x for other p inputs
Gxy_p = Zyy - np.matmul(Zy1, np.matmul(np.linalg.inv(Z11), Z1y))

Gyy_p = Gxy_p[:, 0, 0]
G22_p = Gxy_p[:, 1, 1]
G2y_p = Gxy_p[:, 1, 0]

H2y_miso_2 = G2y_p / G22_p
C2y_p = abs(G2y_p)**2 / (G22_p * Gyy_p)

# ------------------------------------------------------------------------------

# Plotting
fig = plt.figure()

ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,3)
ax3 = fig.add_subplot(2,2,2)
ax4 = fig.add_subplot(2,2,4)


ax1.set_title(r'$Y / X_{1}$')
ax1.semilogx(f, 20 * np.log10(abs(H1y)), label='Analytical')
ax1.semilogx(f_est, 20 * np.log10(abs(H1y_siso_est)), label='SISO')
ax1.semilogx(f_est, 20 * np.log10(abs(H1y_miso_est)), label='MISO 1', linewidth=3)
ax1.semilogx(f_est, 20 * np.log10(abs(H1y_miso_2)), label='MISO 2', linestyle='--')
# ax1.semilogx(f_id, 20 * np.log10(abs(Tc)), 'x')
# ax1.semilogx(f, 20 * np.log10(abs(H1y_id)), label='TF ID' )
# ax1.semilogx(f, 20 * np.log10(abs(H1y_ss_id)), label='SS ID' )

ax2.set_title('Coherence')
ax2.semilogx(f_est, C1y, label='Ordinary coherence')
ax2.semilogx(f_est, C1y_p, label='Partial coherence')
ax2.set_ylim((0, 1.1))

ax3.set_title(r'$Y / X_{2}$')
ax3.semilogx(f, 20 * np.log10(abs(H2y)), label='Analytical')
ax3.semilogx(f_est, 20 * np.log10(abs(H2y_siso_est)), label='SISO')
ax3.semilogx(f_est, 20 * np.log10(abs(H2y_miso_est)), label='MISO 1', linewidth=3)
ax3.semilogx(f_est, 20 * np.log10(abs(H2y_miso_2)), label='MISO 2', linestyle='--')
# ax3.semilogx(f_id, 20 * np.log10(abs(Tc)), 'x')
# ax3.semilogx(f, 20 * np.log10(abs(H1y_id)), label='TF ID' )
# ax3.semilogx(f, 20 * np.log10(abs(H1y_ss_id)), label='SS ID' )

ax4.set_title('Coherence')
ax4.semilogx(f_est, C2y, label='Ordinary coherence')
ax4.semilogx(f_est, C2y_p, label='Partial coherence')
ax4.set_ylim((0, 1.1))

ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()

ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()

plt.show()