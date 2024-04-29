'''
Making sense of weighted sums as integral approximations.

Created by: simmeon
Last Modified: 28/04/24
License: MIT

'''

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')

def f(t):
    return t**3 -2*t**2 - t + 3

# Perform a weighted sum approximation
N = 10  # higher number of samples, N, gives a better approximation
sum = 0
t0 = 0
h = 2
nodes = np.linspace(t0, t0+h, N)

for node in nodes:
    sum += f(node)
weighted_sum = sum / N

print('Weighted sum area: ', weighted_sum * h)

# Plotting
t = np.arange(t0, t0 + h, 0.01)
y = f(t)

# Integration result
plt.subplot(1,2,1)
plt.plot(t,y)
plt.fill_between(t, y, alpha=0.3)
plt.annotate('Area = 2.67', (1,2.5), fontsize=20)
plt.title('Area from integration', fontsize=20)

# Weighted sum result
plt.subplot(1,2,2)
plt.plot(t,y)
plt.fill_between(t, weighted_sum, alpha=0.3)
plt.annotate('Area = {:.2f}'.format(weighted_sum * h), (1,2.5), fontsize=20)
plt.title('Area from weighted sum', fontsize=20)

plt.show()