# create a random 3 by 3 matrix and calculate z score columnwise, print matrix before and after zscores

import numpy as np
import random

matrix = np.random.rand(3, 3)
print(matrix)
mean = np.mean(matrix, axis=0)
std_dev = np.std(matrix, axis=0)
print(mean)
print(matrix - mean)
z_scores = (matrix - mean) / std_dev
print(z_scores)