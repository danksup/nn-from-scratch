import numpy as np

a = [1,2,3,4,5,6,7]
b = np.lib.stride_tricks.sliding_window_view(a, 2)
print(b)