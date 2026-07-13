import numpy as np
import mlx.core as mx

E = 3
a = mx.array([1,1,2,0,1,1,2], dtype=mx.int32)
b = mx.zeros(E)
b = b.at[a].add(1)
print(b)