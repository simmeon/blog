'''
Implementing and comparing the Runge-Kutta 2nd and 4th order methods.
We can also compare to the Euler method (1st order).

Created by: simmeon
Last Modified: 29/04/24
License: MIT

'''

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')

# Define some functions to test our solvers with
def dydt(t, y):
    return 3 * t**2 + 20 * np.cos(10 * t)

def analytical_solution(t):
    return t**3 + 2 * np.sin(10 * t)  # Analytical solution to the ODE 

# # Define our cooling function
# T0 = 30
# T_surr = 20
# k = 1
# def analytical_solution(t):
#     return T_surr + (T0 - T_surr)*np.exp(-k*t)

# # Define the derivative (only depends on the current temp, not time)
# def dydt(t,y):
#     return -k * (y - T_surr)

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

    return y

def rk2(dydt, t0, y0, t_end, h):
    """
    Solves a first-order ODE using the 2nd order Runge-Kutta method

    Args:
        dydt: The derivative function to integrate
        t0: Initial value of time
        y0: Initial value of y
        t_end: Final time for integration
        h: Step size

    Returns:
        An array of y values for each time step
    """
    t = np.arange(t0, t_end+h, h)  # Array of time points
    y = np.zeros_like(t)
    y[0] = y0

    for i in range(1, len(t)):
        k1 = h * dydt(t[i - 1], y[i - 1])
        k2 = h * dydt(t[i - 1] + h, y[i - 1] + k1)
        y[i] = y[i - 1] + (k1 + k2) / 2

    return y

def rk4(dydt, t0, y0, t_end, h):
    """
    Solves a first-order ODE using the 4th order Runge-Kutta method

    Args:
        dydt: The derivative function to integrate
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
        k1 = h * dydt(t[i - 1], y[i - 1])
        k2 = h * dydt(t[i - 1] + h / 2, y[i - 1] + k1 / 2)
        k3 = h * dydt(t[i - 1] + h / 2, y[i - 1] + k2 / 2)
        k4 = h * dydt(t[i], y[i - 1] + k3) 

        y[i] = y[i - 1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
 
    return y

# Define simulation parameters
t0 = 0
y0 = 0
t_end = 5
h = 0.1

# Solve using both methods
y_euler = euler_solver(dydt, t0, y0, t_end, h)
y_rk2 = rk2(dydt, t0, y0, t_end, h)
y_rk4 = rk4(dydt, t0, y0, t_end, h)

# Get analytical solution
t = np.arange(t0, t_end+h, h)  # Array of time points
y_analytical = analytical_solution(t)

# Calculate errors
error_euler = np.abs(y_euler - y_analytical)
error_rk2 = np.abs(y_rk2 - y_analytical)
error_rk4 = np.abs(y_rk4 - y_analytical)

# Plot the numerical solutions
plt.subplot(1,2,1)
plt.plot(t, y_euler, label='Euler')
plt.plot(t, y_rk2, label='RK2')
plt.plot(t, y_rk4, label='RK4')
plt.plot(t, y_analytical, label='Analytical', linestyle='--') 
plt.xlabel('Time [s]', fontsize=20)
plt.ylabel('y(t)', fontsize=20)
plt.title('Numerical Solutions', fontsize=20)
plt.legend(fontsize=15)

# Plot the errors
plt.subplot(1,2,2)
plt.plot(t, error_euler, label='Euler')
plt.plot(t, error_rk2, label='RK2')
plt.plot(t, error_rk4, label='RK4')
plt.xlabel('Time [s]', fontsize=20)
plt.ylabel('Abs Error', fontsize=20)
plt.title('Absolute Errors', fontsize=20)
plt.legend(fontsize=15)

plt.show()