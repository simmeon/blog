'''
Some sample functions of our digital signal stationary random process.
Created by: simmeon
Last Modified: 2025-01-25
License: MIT

'''

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')

num_bits = 10

# Calculate random series of bits
bits1 = np.random.randint(0, 2, num_bits)
bits2 = np.random.randint(0, 2, num_bits)
bits3 = np.random.randint(0, 2, num_bits)

N = 100 # Number of sample points per bit

# Make length of each bit 1 s
t = np.linspace(0, num_bits, num_bits * N)

# sample functions
x1 = []
x2 = []
x3 = []

# Make sample function have N points per bit and add offsets
for bit in bits1:
    x1 = x1 + [bit] * N

for bit in bits2:
    x2 = x2 + [bit] * N
x2 = x2[35:] + x2[:35] # offset

for bit in bits3:
    x3 = x3 + [bit] * N
x3 = x3[60:] + x3[:60] # offset


# Plotting
fig = plt.figure()

ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)

ax1.set_title('Digital Signal Sample Functions', fontsize=20)

ax1.set_xticks(np.arange(0, 11, 1))
ax2.set_xticks(np.arange(0, 11, 1))
ax3.set_xticks(np.arange(0, 11, 1))

ax1.grid()
ax2.grid()
ax3.grid()

ax1.grid(visible=True, axis='x', which='both')
ax2.grid(visible=True, axis='x', which='both')
ax3.grid(visible=True, axis='x', which='both')

ax1.plot(t, x1)
ax2.plot(t, x2)
ax3.plot(t, x3)

ax2.set_ylabel('Voltage (V)', fontsize=20)
ax3.set_xlabel('Time (s)', fontsize=20)

plt.show()