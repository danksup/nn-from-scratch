import numpy as np
import mlx.core as mx

# a = [[1,2,3], [4,5,6], [7,8,9]]
# index = [2,3,4]
# b = np.triu(a)
# c = np.where(b,)
# print(b)

a =mx.array( [[[1,2,3], [4,5,6], [7,8,9]],[[1,2,3], [4,5,6], [7,8,9]]] )
# idxs = mx.array([[2],[0],[1]])

# b = a.at[0,1,1].add(-1)
# a[:,:,:2] -= 1
# print(a)
c = mx.zeros(a.shape[-1])
# print(c)
b = mx.array([2,3,4])
# print(a.shape)

b = np.unique(a, return_counts=True)
# print(type(b[0]))

m =mx.array( [
                [
                    [1,2,3,4], 
                    [4,5,6,7], 
                    [7,8,9,10] 
                    
                ] 
            ])
ax0 = mx.zeros(m.shape[1],dtype=mx.int32)
ax1 = mx.arange(m.shape[1])
idx = mx.array([0,], dtype=mx.int32)
m = m.at[..., idx].add(1)
print(m)
