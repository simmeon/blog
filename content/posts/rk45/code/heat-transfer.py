'''
Plotting the temperature of a point with different starting temperatures.

Created by: simmeon
Last Modified: 28/04/24
License: MIT

'''

import matplotlib.pyplot as plt
import numpy as np

# Define system parameters
T_surr = 20     # surrounding temperature of 20 C
k = 1           # proportionality constant of 1 for simplicity

# Define our cooling function
def solve_T(t, T0):
    return T_surr + (T0 - T_surr)*np.exp(-k*t)

# Create an array of times from 0-5 seconds
t = np.arange(0, 5, 0.1)

# Solve and plot the solutions for different initial values of T
T0_choices = [30, 25, 20, 15, 10]

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')

for T0 in T0_choices:
    T = solve_T(t, T0)
    plt.plot(t, T, label='T0 = {} C'.format(T0))

plt.title('Temperature over time', fontsize=20)
plt.xlabel('Time [s]', fontsize=20)
plt.ylabel('Temperature [C]', fontsize=20)
plt.xlim(0, 5)
plt.ylim(10, 30)
plt.grid(alpha=0.7)
plt.legend()
plt.show()



