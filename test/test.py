import numpy as np
import mlx.core as mx

a = mx.array([True, True, False], dtype=mx.bool_)
b = mx.array(a, mx.int32)
print(b)