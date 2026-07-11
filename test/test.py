import numpy as np
import mlx.core as mx

E = 3
T = 4

a = mx.array([1,1,2,2,3,3,3], mx.int32)
b = np.pad(a, (1,0), 'constant')
c = mx.pad(a, (1,0), 'constant')
print(b)
print(c)
# a[0] = 19
# print(a)
# # count = mx.zeros(4, mx.int32)
# # count = count.at[a].add(1)

# # cum = mx.cumsum(count)
# # rows = mx.arange(cum.shape[0])
# # print(cum)