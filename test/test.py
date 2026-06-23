from engine.backend import Backend
import numpy as np
import mlx.core as mlx
from engine.activations import softmax

nx = Backend("numpy")

f = mlx.array([2,3,4], mlx.float32)
a = mlx.array([2,3,4], mlx.float32)
# g = nx.dot(nx.array(f),nx.array(a))

p = [0.9, 0.4, 0.5]
cdf = np.cumsum(p)
r = np.random.uniform()
idx = np.argmax(cdf >= r)
b = a[idx]
print(b)
# a = f @ a
# i = mlx.tensordot(mlx.array(f),mlx.array(a), axes=1)
# print(a)


