import numpy as np

y0 = [[1, 2]]

y = np.array(y0)

b = np.array([[4,7]])

y = np.append(y, b, axis=0)

print(y)