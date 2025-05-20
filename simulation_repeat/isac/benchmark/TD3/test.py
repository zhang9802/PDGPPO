import numpy as np

bins = np.linspace(-1, 1, 3)
print(bins)
indexes = np.digitize(-0.5, bins) - 1
print(indexes)