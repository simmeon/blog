'''
Performing Euler integration to solve the cooling equation. We will compare with the analytical solution.

Created by: simmeon
Last Modified: 28/04/24
License: MIT

'''

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')

# Define system
T_surr = 20     # surrounding temperature of 20 C
k = 1           # proportionality constant of 1 for simplicity

# Define our cooling function
def solve_T(t, T0):
    return T_surr + (T0 - T_surr)*np.exp(-k*t)

# Define the derivative (only depends on the current temp, not time)
def dTdt(T):
    return -k * (T - T_surr)

# Define our Euler integration function
def euler(dTdt, T0, t0, t_end, h=0.1):
    '''
    Takes the derivative function, an initial condition, the time we want to integrate until,
    and a step size.

    Returns arrays of time and temperature values
    '''
    t = np.arange(t0, t_end+h, h)
    T = np.zeros(len(t))

    T[0] = T0

    for i in range(1,len(t)):
        T[i] = T[i-1] + h * dTdt(T[i-1])

    return t, T


# Parameters
t0 = 0
T0 = 30
t_end = 5

# Get analytical solution
t_analytical = np.arange(t0, t_end+0.1, 0.1)
T_analytical = solve_T(t_analytical, T0)

# Get Euler numerical solution
t_euler_05, T_euler_05 = euler(dTdt, T0, t0, t_end, 0.5)
t_euler_01, T_euler_01 = euler(dTdt, T0, t0, t_end, 0.1)

# Plotting
plt.plot(t_analytical, T_analytical, label='Analytical', linestyle='--')
plt.plot(t_euler_05, T_euler_05, label='Euler method, h = 0.5')
plt.plot(t_euler_01, T_euler_01, label='Euler method, h = 0.1')

plt.title("Analytical and Euler method solutions to Newton's cooling law", fontsize=20)
plt.xlabel('Time [s]', fontsize=20)
plt.ylabel('Temperature [C]', fontsize=20)

plt.legend(fontsize=15)
plt.show()