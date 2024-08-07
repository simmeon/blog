'''
Implementing an adaptive step size by doubling or halving the step size 
depending on the error.

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


def euler_step(dydt, t0, y0, h):
    """
    Takes a single 1st order integration step and returns the y value.

    Args:
        dydt: The differential equation (dy/dt) as a Python function 
        t0: Initial value of time
        y0: Initial value of y
        h: Step size

    Returns:
        The y value after the step.

    """

    y = y0 + h * dydt(t0, y0)

    return y

def rk2_step(dydt, t0, y0, h):
    """
    Takes a single 2nd order integration step and returns the y value.

    Args:
        dydt: The differential equation (dy/dt) as a Python function 
        t0: Initial value of time
        y0: Initial value of y
        h: Step size

    Returns:
        The y value after the step.

    """

    k1 = h * dydt(t0, y0)
    k2 = h * dydt(t0 + h, y0 + k1)
    y = y0 + (k1 + k2) / 2

    return y


# Our adaptive Euler numerical solver
def euler_solver(dydt, t0, y0, t_end, tol):
    """
    Solves a first-order ODE using the Euler method with adaptive
    step sizing.

    Args:
        dydt: The differential equation (dy/dt) as a Python function 
        t0: Initial value of time
        y0: Initial value of y
        t_end: Final time for integration
        tol: Error tolerance

    Returns:
        Arrays of y, t and local error values for each time step
    """
    h = 0.1 # set some initial step size
    t = [t0]
    y_euler = [y0]
    y_rk2 = [y0]
    error = [0]
    
    i = 1
    while t[-1] < t_end:
        # Take a Euler step
        y_euler.append(euler_step(dydt, t[i-1], y_euler[i-1], h))
        t.append(t[i-1] + h)

        # Take a RK2 step, starting from the previous Euler step value
        y_rk2.append(rk2_step(dydt, t[i-1], y_euler[i-1], h))

        isStepGood = False

        error.append(np.abs(y_rk2[i] - y_euler[i]))

        if error[i] < tol:
            # accept step, make step size bigger for next step
            isStepGood = True
            h = h * 2

        while not isStepGood:
            # Take a step with both methods
            y_euler[i] = euler_step(dydt, t[i-1], y_euler[i-1], h)
            t[i] = t[i-1] + h
            y_rk2[i] = rk2_step(dydt, t[i-1], y_euler[i-1], h)

            # Calculate the error between the methods
            error[i] = np.abs(y_rk2[i] - y_euler[i])

            # Check error to accept or reject the step
            if error[i] > tol:
                # reject step, halve step size
                h = h / 2
            else:
                # accept step, make step size bigger for next step
                isStepGood = True
                h = h * 2

        i += 1

    return t, y_euler, error


# Define simulation parameters
t0 = 0
y0 = 0
t_end = 2
tol = 1e-2

# Numerically integrate
t_euler, y_euler, local_error = euler_solver(dydt, t0, y0, t_end, tol)

t_euler = np.array(t_euler)
y_euler = np.array(y_euler)

# Get analytical solution
t_analytical = np.arange(t0, t_end, 0.001)
y_analytical = analytical_solution(t_analytical)

# Get error
y_analytical_compare = analytical_solution(t_euler)
error = np.abs(y_euler - y_analytical_compare)

# Plotting
plt.subplot(3,1,1)
plt.plot(t_analytical, y_analytical, label='Analytical', linestyle='-')
plt.plot(t_euler, y_euler, label='Euler method')
plt.scatter(t_euler, y_euler, s=30, marker='o', facecolors='none', color='C1')

plt.title("Integrating with a adaptive step size, tol = 0.01", fontsize=20)
plt.ylabel('y', fontsize=20)
plt.legend(fontsize=15)
plt.text(0, -0.75, f"Number of steps: {len(t_euler)}", fontsize=15)

plt.subplot(3,1,2)
plt.plot(t_euler, error)
plt.ylabel("absolute error", fontsize=20)
plt.xlabel('t', fontsize=20)

plt.subplot(3,1,3)
plt.plot(t_euler, local_error)
plt.ylabel("local step error", fontsize=20)
plt.xlabel('t', fontsize=20)

plt.show()
