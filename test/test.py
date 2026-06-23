from engine.backend import Backend
import numpy as np
from engine.activations import softmax

nx = Backend("mlx")

# a = nx.float_32(333)
# a +=  nx.float_32(333)

# a = nx.ones(3)
# b = nx.array([44,2,5])

a = [2,3,4,2]
a = softmax(a)
print(a)

# a = np.exp([2,3,4,2])
# b = np.exp(a)
# print(b)
# nx.add_at(b,a,4)
# print(nx.add_at(b,a,4))


