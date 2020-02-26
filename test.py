import numpy as np
import nn_lib

x = np.array([[100, 10, 5],
              [300, 5, 0.35],
              [500, 7, 0.09]])

layer = nn_lib.SigmoidLayer()
print(layer.forward(x))

# y = np.array([1, 1, 2])
# x_normed = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
# print(x.max(axis=0))
# print(x)
# print(x_normed)
# print(x + y)
