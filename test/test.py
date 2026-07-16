import numpy as np
import mlx.core as mx
from typing import Any
E = 4
top_k = 2
a =[
  [  # Batch 0

    [  # Head 0
      [ 1,  2,  3,  4],
      [ 5,  6,  7,  8],
      [ 9, 10, 11, 12]
    ],

    [  # Head 1
      [13, 14, 15, 16],
      [17, 18, 19, 20],
      [21, 22, 23, 24]
    ]

  ],

  [  # Batch 1

    [  # Head 0
      [25, 26, 27, 28],
      [29, 30, 31, 32],
      [33, 34, 35, 36]
    ],

    [  # Head 1
      [37, 38, 39, 40],
      [41, 42, 43, 44],
      [45, 46, 47, 48]
    ]

  ]
]
a = mx.array(a)
B,n_kv_heads,T,_ = a.shape
w = 2
n = a.shape[-1]
pad = [(0,0), (0,0),(w-1,0), (0,0)]
b = mx.pad(a, pad)
P = T + w - 1
shape = a.shape[0], a.shape[1], a.shape[2], w, a.shape[3]
stride = a.shape[-1] * n_kv_heads * P, a.shape[-1] * P, a.shape[-1], a.shape[-1], 1
c = mx.as_strided(b, shape=shape, strides= stride)
print(b)
print("\n")
print(c)
print(c.shape)
print(stride)
