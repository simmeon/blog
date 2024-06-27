'''
Implementing a complete RK45 algorithm.

Created by: simmeon
Last Modified: 19/05/24
License: MIT

'''

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')

# Define function (for analytical solution) and its derivative
def dydt(t, y):
    return 5*t**4 * np.cos(t**5)

def analytical_solution(t):
    return np.sin(t**5)


# ----- Dormand-Prince coefficients ----- #

# Alpha in notation, coefficients describe what percentage of the full
# step we evaluate each derivative at
A = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])

# Beta in notation, coefficients describe how much of each previous change in y
# we add to the current estimate
B = np.array([
        [0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
    ])

# Weighting for each derivative in the approximation, w in notation
W = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])

# Coefficients for error calculation
E = np.array([-71/57600, 0, 71/16695, -71/1920, 17253/339200, -22/525, 1/40])

# ------------------------------------------------------------------ #

# Define a function to take a single RK45 step
# To improve efficiency, k7 can be reused as k1 in the following step
def rk45_step(dydt, t0, y0, h):
    """
    Takes a single 5th order integration step and returns the 
    4th order step along with the 5th order error.

    Args:
        dydt: The differential equation (dy/dt) as a Python function 
        t0: Initial value of time
        y0: Initial value of y
        h: Step size

    Returns:
        The the 4th order y value along with the 5th order error.

    """

    k1 = h * dydt(t0, y0)
    k2 = h * dydt(t0 + A[1] * h, y0 + B[1][0]*k1)
    k3 = h * dydt(t0 + A[2] * h, y0 + B[2][0]*k1 + B[2][1]*k2)
    k4 = h * dydt(t0 + A[3] * h, y0 + B[3][0]*k1 + B[3][1]*k2 + B[3][2]*k3)
    k5 = h * dydt(t0 + A[4] * h, y0 + B[4][0]*k1 + B[4][1]*k2 + B[4][2]*k3 + B[4][3]*k4)
    k6 = h * dydt(t0 + A[5] * h, y0 + B[5][0]*k1 + B[5][1]*k2 + B[5][2]*k3 + B[5][3]*k4 + B[5][4]*k5)

    y = y0 + W[0]*k1 + W[1]*k2 + W[2]*k3 + W[3]*k4 + W[4]*k5 + W[5]*k6

    k7 = h * dydt(t0 + A[6] * h, y)

    error = np.abs( E[0]*k1 + E[1]*k2 + E[2]*k3 + E[3]*k4 + E[4]*k5 + E[5]*k6 + E[6]*k7 )

    return y, error

# Define the full RK45 solver function
def rk45_solver(dydt, t0, y0, t_end, tol):
    """
    Solves a first-order ODE using the RK45 method.

    Args:
        dydt: The differential equation (dy/dt) as a Python function 
        t0: Initial value of time
        y0: Initial value of y
        t_end: Final time for integration
        tol: Error tolerance

    Returns:
        Arrays of y, t and local error values for each time step
    """
    h = 1e-3 # set some initial step size
    t = [t0]
    y = [y0]
    error = [0]
    
    i = 1
    while t[-1] < t_end:
        # Take a step
        t.append(t[i-1] + h)
        y_step, error_step = rk45_step(dydt, t[i-1], y[i-1], h)
        y.append(y_step)
        error.append(error_step)

        # Update step size after step
        h = 0.9 * h * (tol / error[i]) ** 0.2

        isStepGood = False

        if error[i] < tol:
            # accept step
            isStepGood = True

        while not isStepGood:
            # If there was too much error...

            # Take a step
            t[i] = t[i-1] + h # update our time value with the new step size
            y_step, error_step = rk45_step(dydt, t[i-1], y[i-1], h)
            y[i] = y_step
            error[i] = error_step
            
            # Update step size after step
            h = 0.9 * h * (tol / error[i]) ** 0.2

            # Check error to accept or reject the step
            if error[i] < tol:
                isStepGood = True

        i += 1

    return t, y, error


# Define simulation parameters
t0 = 0
y0 = 0
t_end = 3
tol = 1e-12

# Numerically integrate
t_rk45, y_rk45, local_error = rk45_solver(dydt, t0, y0, t_end, tol)

t_rk45 = np.array(t_rk45)
y_rk45 = np.array(y_rk45)

# Get analytical solution
t_analytical = np.arange(t0, t_end, 0.001)
y_analytical = analytical_solution(t_analytical)

# Get error
y_analytical_compare = analytical_solution(t_rk45)
error = np.abs(y_rk45 - y_analytical_compare)

# Plotting
plt.subplot(3,1,1)
plt.plot(t_analytical, y_analytical, label='Analytical', linestyle='-')
plt.plot(t_rk45, y_rk45, label='RK45 method')
#plt.scatter(t_rk45, y_rk45, s=30, marker='o', facecolors='none', color='C1')

plt.title(f"RK45 Solver, tol = {tol}", fontsize=20)
plt.ylabel('y', fontsize=20)
plt.legend(fontsize=15)
plt.text(0, -0.75, f"Number of steps: {len(t_rk45)}", fontsize=15)

plt.subplot(3,1,2)
plt.plot(t_rk45, error)
plt.ylabel("absolute error", fontsize=20)

plt.subplot(3,1,3)
plt.plot(t_rk45, local_error)
plt.ylabel("local step error", fontsize=20)
plt.xlabel('t', fontsize=20)

plt.show()