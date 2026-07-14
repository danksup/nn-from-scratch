import numpy as np
import mlx.core as mx

E = 4
top_k = 2
a = mx.array([[[29,12,3,1],[29,12,30,1],[29,120,3,1],[21,12,3,1]], [[29,12,3,1],[29,12,30,1],[29,120,3,1],[21,12,3,1]]], dtype=mx.int32)
print(mx.sum(a, 1).shape)