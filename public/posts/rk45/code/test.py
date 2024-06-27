 1'''
 2Plotting the temperature of a point with different starting temperatures.
 3
 4Created by: simmeon
 5Last Modified: 28/04/24
 6License: MIT
 7
 8'''
 9
10import matplotlib.pyplot as plt
11import numpy as np
12
13# Define system parameters
14T_surr = 20     # surrounding temperature of 20 C
15k = 1           # proportionality constant of 1 for simplicity
16
17# Define our cooling function
18def solve_T(t, T0):
19    return T_surr + (T0 - T_surr)*np.exp(-k*t)
20
21# Create an array of times from 0-5 seconds
22t = np.arange(0, 5, 0.1)
23
24# Solve and plot the solutions for different initial values of T
25T0_choices = [30, 25, 20, 15, 10]
26
27plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')
28
29for T0 in T0_choices:
30    T = solve_T(t, T0)
31    plt.plot(t, T, label='T0 = {} C'.format(T0))
32
33plt.title('Temperature over time', fontsize=20)
34plt.xlabel('Time [s]', fontsize=20)
35plt.ylabel('Temperature [C]', fontsize=20)
36plt.xlim(0, 5)
37plt.ylim(10, 30)
38plt.grid(alpha=0.7)
39plt.legend()
40plt.show()
