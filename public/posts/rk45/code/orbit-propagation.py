'''
An application of the RK45 ODE solver. We will solve for the position
of a satellite around Earth given an initial position and velocity.

Created by: simmeon
Last Modified: 19/05/24
License: MIT

'''

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')

# Define function derivative
def dxdt(t, x):
    r_mag = np.sqrt(x[0]**2 + x[1]**2)
    mu = 398600.4415;   # [km^3 / (kg*s^2)]

    return np.array([x[2], x[3], -mu * x[0] / (r_mag**3), -mu * x[1] / (r_mag**3)])



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
#W = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])

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
    y = np.array([y0])
    error = [0]
    
    i = 1
    while t[-1] < t_end:
        # Take a step
        t.append(t[i-1] + h)
        y_step, error_step = rk45_step(dydt, t[i-1], y[i-1], h)
        y = np.append(y, np.array([y_step]), axis=0)
        error.append(error_step)

        # Update step size after step
        h = 0.9 * h * (tol / max(error[i])) ** 0.2

        isStepGood = False

        if max(error[i]) < tol:
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
            h = 0.9 * h * (tol / max(error[i])) ** 0.2

            # Check error to accept or reject the step
            if max(error[i]) < tol:
                isStepGood = True

        i += 1

    return t, y, error


# Define orbit parameters
mu = 398600.4415
rp = 6678
e = 0.9
a = rp/(1-e)
T = 2*np.pi*np.sqrt(a**3/mu)

# Define simulation parameters
t0 = 0
x0 = np.array([rp, 0, 0, np.sqrt(2*mu/rp - mu/a)])
t_end = T
tol = 1e-12

# Numerically integrate
t_rk45, y_rk45, local_error = rk45_solver(dxdt, t0, x0, t_end, tol)
y_rk45 = np.array(y_rk45)

print(len(t_rk45))

# Plotting
plt.subplot(2,1,1)
plt.plot(y_rk45[:, 0], y_rk45[:, 1])
plt.axis('equal')
plt.scatter(0, 0, s=50) # dot for Earth
plt.title("Satellite Orbit Around Earth", fontsize=20)
plt.ylabel('y [km]', fontsize=20)
plt.xlabel('x [km]', fontsize=20)

plt.subplot(2,1,2)
plt.plot(t_rk45, np.sqrt(y_rk45[:, 2]**2 + y_rk45[:, 3]**2), color='C1')
plt.title("Satellite Velocity", fontsize=20)
plt.ylabel('velocity [km/s]', fontsize=20)
plt.xlabel('t [s]', fontsize=20)

plt.show()