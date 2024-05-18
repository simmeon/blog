'''
Seeing how we can improve our solution and efficiency by using two different 
step sizes at different times.

Created by: simmeon
Last Modified: 18/05/24
License: MIT

'''

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')

# Define function and its derivative
def dydt(t, y):
    return 5*t**4 * np.cos(t**5)

def analytical_solution(t):
    return np.sin(t**5)

# Our Euler numerical solver
def euler_solver(dydt, t0, y0, t_end, h):
    """
    Solves a first-order ODE using the Euler method

    Args:
        dydt: The differential equation (dy/dt) as a Python function 
        t0: Initial value of time
        y0: Initial value of y
        t_end: Final time for integration
        h: Step size

    Returns:
        An array of y values for each time step
    """
    t = np.arange(t0, t_end+h, h)  
    y = np.zeros_like(t)
    y[0] = y0

    for i in range(1, len(t)):
        y[i] = y[i - 1] + h * dydt(t[i - 1], y[i - 1])

    return t, y


# Define simulation parameters
t0 = 0
y0 = 0
t_end1 = 1
t_end2 = 2

h1 = 0.05 # first, bigger step size
h2 = 0.01 # second. smaller step size

# Numerically integrate
t_euler1, y_euler1 = euler_solver(dydt, t0, y0, t_end1, h1)
t_euler2, y_euler2 = euler_solver(dydt, t_end1, y_euler1[-1], t_end2, h2)

y_euler = np.concatenate((y_euler1, y_euler2))
t_euler = np.concatenate((t_euler1, t_euler2))

# Get analytical solution
t_analytical = np.arange(t0, t_end2+h2, 0.001)
y_analytical = analytical_solution(t_analytical)

# Get error
y_analytical_compare = analytical_solution(t_euler)
error = np.abs(y_euler - y_analytical_compare)

# Plotting
plt.subplot(2,1,1)
plt.plot(t_analytical, y_analytical, label='Analytical', linestyle='-')
plt.plot(t_euler, y_euler, label='Euler method')
plt.vlines(t_end1, -1, 1, colors='white', alpha=0.5)

plt.title("Error when integrating with two different step sizes", fontsize=20)
plt.ylabel('y', fontsize=20)
plt.legend(fontsize=15)

plt.subplot(2,1,2)
plt.plot(t_euler, error)
plt.vlines(t_end1, 0, 0.5, colors='white', alpha=0.5)
plt.ylabel("absolute error", fontsize=20)
plt.xlabel('t', fontsize=20)

plt.show()