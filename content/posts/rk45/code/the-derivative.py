'''
Visualising how the derivative can be used to estimate the value of a function.

Created by: simmeon
Last Modified: 28/04/24
License: MIT

'''

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')

# Define some function we want to find the value of
def func(t):
    return t**2

# Define the derivative line of the function
def dydt(t, t0, y0):
    m = 2 * t0
    return m * (t-t0) + y0

# Create an array of time values to plot the function
t_func = np.arange(0.5, 2, 0.1)

# Estimation parameters
t0 = 1.0                    # known initial value
y0 = func(t0)               # known initial value
h = [1, 0.5, 0.2]           # step size options

# Plotting
# Find and plot function values
y_func = func(t_func)
plt.plot(t_func, y_func)
plt.plot(t0, y0, 'x', color='C0')

# Plot the derivative line for each step size
colours = ['C1', 'C2', 'C3']
for i in range(len(h)):
    t_dydt = np.arange(t0, t0 + h[i], 0.1)
    y_dydt = dydt(t_dydt, t0, y0)
    plt.plot(t_dydt, y_dydt, color=colours[i], label='Step = {:.1f}'.format(h[i]))
    plt.plot(t_dydt[-1], y_dydt[-1], 'o', color=colours[i])

# Show plot
plt.title("Estimating a function by using the derivative", fontsize=20)
plt.legend(loc='lower right', fontsize=10)
plt.show()