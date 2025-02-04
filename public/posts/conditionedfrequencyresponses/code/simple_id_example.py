"""
A simple two-input (partically correlated), one-output linear system to test
identification with.

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
output_noise_sd = 0.00

# System coefficienets
a = 2
b = 3
c = 2
d = 0.0
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

# Define input x2: no particular reason I made it be this, just trying to roughly cover freq range
x2_uc = np.sin(2*pi*0.2*t) + np.sin(2*pi*0.37*t) + np.sin(2*pi*1.2*t) + np.sin(2*pi*1.7*t) + np.sin(2*pi*2.0*t) + np.sin(2*pi*2.6*t) + np.sin(2*pi*3.0*t) + np.sin(2*pi*3.3*t) + np.sin(2*pi*4.2*t)
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
# y = y + np.random.normal(0, output_noise_sd, len(y))


# ------------------------------------------------------------------------------

# Frequency Responses

# Analytical transfer functions
f = np.logspace(np.log10(fmin), np.log10(fmax), N)
s = 1.0j * 2 * pi * f

H1y = a / (s**2 + c * s + k)
H2y = b / (s**2 + c * s + k)

# SISO Estimated freq response
overlap = 0.8

f_est, Gxx_matrix, Gyy_matrix, Gxy_matrix = frequency_response.get_all_spectral_densities([x1, x2], [y], overlap, T, fs, fmin, fmax)
H1y_siso_est = Gxy_matrix[0,0] / Gxx_matrix[0,0]
H2y_siso_est = Gxy_matrix[1,0] / Gxx_matrix[1,1]
C1y = abs(Gxy_matrix[0, 0])**2 / (Gxx_matrix[0, 0].real * Gyy_matrix[0, 0].real)
C2y = abs(Gxy_matrix[1, 0])**2 / (Gxx_matrix[1, 1].real * Gyy_matrix[0, 0].real)

# Swap axes so frequency is first index
Gxy_matrix = np.swapaxes(Gxy_matrix, 0, 2)
Gxy_matrix = np.swapaxes(Gxy_matrix, 1, 2)

Gxx_matrix = np.swapaxes(Gxx_matrix, 0, 2)
Gxx_matrix = np.swapaxes(Gxx_matrix, 1, 2)

# MISO conditioned frequency response
Hxy_miso_est = []
for f_idx in range(len(f_est)):
    Hxy_miso_est.append(np.linalg.solve(Gxx_matrix[f_idx], Gxy_matrix[f_idx]))
Hxy_miso_est = np.array(Hxy_miso_est)

H1y_miso_est = Hxy_miso_est[:, 0, 0]
H2y_miso_est = Hxy_miso_est[:, 1, 0]

# ------------------------------------------------------------------------------
# Identifying transfer function with numerical optimization

# Frequencies to use for optimization (~20)
# NOTE: these should be chosen based on freq range where coherence is high (> 0.6)
fmin_id = 1.0
fmax_id = 3
wmin = 2 * pi * fmin_id
wmax = 2 * pi * fmax_id
n_w = 20 # number of points to evaluate at
w_id = np.logspace(np.log10(wmin), np.log10(wmax), n_w)
f_id = np.logspace(np.log10(fmin_id), np.log10(fmax_id), n_w)

# Define transfer function model structure
def T_model(s, x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]

    return x1 / (s ** 2 + x2 * s + x3)

# TF Cost function
# Magnitude (dB) has a weighting of 1.0
# Phase (deg) has weighting of 0.01745
# See Tischler for reasoning
def J(x, Wc, Tc):
    T = T_model(1.0j * w_id, x)
    return 20 / n_w * sum(Wc * (1.0 * (20*np.log10(abs(Tc)) - 20*np.log10(abs(T))) ** 2 + 0.01745 * (np.rad2deg(np.angle(Tc)) - np.rad2deg(np.angle(T))) ** 2))

# Calculate coherence weighting
C1y_new = np.interp(f_id, f_est, C1y)
Wc1 = (1.58 * (1 - np.exp(-C1y_new))) ** 2
C2y_new = np.interp(f_id, f_est, C2y)
Wc2 = (1.58 * (1 - np.exp(-C2y_new))) ** 2

# Transfer function estimate
Tc = np.interp(f_id, f_est, H1y_miso_est)

# Initial guess
x0 = [1, 1, 100]

# Optimize
sol = minimize(J, x0, args=(Wc1, Tc), method='Nelder-Mead', tol=1e-6)
# print(sol.fun) # values of cost function

# Identified transfer function
H1y_id = T_model(s, sol.x)

# ------------------------------------------------------------------------------

# State space identification

# Define state space model
def state_space_tf(s, x):
    a_id = x[0]
    b_id = x[1]
    c_id = x[2]
    k_id = x[3]

    T_ss = []

    if type(s) is complex:
        # Define model structure
        F = np.array([
            [0, 1], 
            [- k_id, - c_id]
        ])

        G = np.array([
            [0, 0], 
            [a_id, b_id]
        ])

        H0 = np.array([1, 0])

        # Find transfer functions from structure
        tmp2 = s * np.identity(2) - F
        tmp3 = np.matmul(np.linalg.inv(tmp2), G)
        T_ss = np.matmul(H0, tmp3)
    else:
        for s_id in s:
            # Define model structure
            F = np.array([
                [0, 1], 
                [- k_id, - c_id]
            ])

            G = np.array([
                [0, 0], 
                [a_id, b_id]
            ])

            H0 = np.array([1, 0])

            # Find transfer functions from structure
            tmp2 = s_id * np.identity(2) - F
            tmp3 = np.matmul(np.linalg.inv(tmp2), G)
            T_ss.append(np.matmul(H0, tmp3))
        T_ss = np.array(T_ss)
        
    return T_ss

# Define cost function to minimise
def K(x, transfer_functions, Wc, freq_id):
    num_freqs = len(freq_id)

    cost = 0

    for i in range(len(transfer_functions)):
        Tc = transfer_functions[i]

        for j in range(num_freqs):
            f = freq_id[j]
            s_id = 1.0j * 2 * np.pi * f

            T = state_space_tf(s_id, x)

            cost = cost + Wc[i, j] * (1.0 * (20 * np.log10(abs(Tc[j])) - 20 * np.log10(abs(T[i])))**2 + 0.01745 * (np.rad2deg(np.angle(Tc[j])) - np.rad2deg(np.angle(T[i])))**2)

        cost = cost * 20 / num_freqs
    
    return cost

# Inerpolate frequency response estimates to have same freqencies as id points
H1y_for_id = np.interp(f_id, f_est, H1y_miso_est)
H2y_for_id = np.interp(f_id, f_est, H2y_miso_est)


# Initial guess for variables
x0 = [1, 1, 1, 50] # a, b, c, k

# Optimize
sol = minimize(K, x0, args=(np.array([H1y_for_id, H2y_for_id]), np.array([Wc1, Wc2]), f_id), method='Nelder-Mead', tol=1e-6)

# print(sol.x)
# print(f'Cost function: {sol.fun}')


# Get frequency responses from identified state space model
H_ss_id = state_space_tf(s, sol.x)
H1y_ss_id = H_ss_id[:, 0]
H2y_ss_id = H_ss_id[:, 1]


# Plotting
fig = plt.figure()

ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)


ax1.set_title(r'$Y / X_{1}$')
ax1.semilogx(f, 20 * np.log10(abs(H1y)), label='Analytical')
ax1.semilogx(f_est, 20 * np.log10(abs(H1y_siso_est)), label='SISO')
ax1.semilogx(f_est, 20 * np.log10(abs(H1y_miso_est)), label='MISO' )
ax1.semilogx(f_id, 20 * np.log10(abs(Tc)), 'x')
ax1.semilogx(f, 20 * np.log10(abs(H1y_id)), label='TF ID' )
ax1.semilogx(f, 20 * np.log10(abs(H1y_ss_id)), label='SS ID' )

ax2.set_title(r'$Y / X_{2}$')
ax2.semilogx(f, 20 * np.log10(abs(H2y)), label='Analytical')
ax2.semilogx(f_est, 20 * np.log10(abs(H2y_siso_est)), label='SISO')
ax2.semilogx(f_est, 20 * np.log10(abs(H2y_miso_est)), label='MISO' )
ax2.semilogx(f, 20 * np.log10(abs(H2y_ss_id)), label='SS ID' )
# ax2.semilogx(f_est, C1y)

ax1.grid()
ax2.grid()

ax1.legend()
ax2.legend()

plt.show()