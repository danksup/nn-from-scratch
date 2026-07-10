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
c = np.unique(m, return_counts=True)
ctype = getattr(c[0], 'dtype')
print(ctype)

d = mx.array(c[0])
print(d)
