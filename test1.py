import matplotlib.pyplot as plt
import numpy as np

a = np.array([1, 2, 3, 4])
print(a)
a = np.random.rand(100)
print(a)
plt.plot(a, 'r.')
plt.show()