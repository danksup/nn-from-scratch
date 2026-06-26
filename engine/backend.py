from typing import Any
nx: Any
import os
class Backend:
    def __init__(self) -> None:
        self.backend = os.environ.get("USE_BACKEND", "auto").lower()
        self.nx:Any

        if self.backend == "auto":
            try:
                import mlx.core as nx
                self.nx = nx
                self.backend = "MLX"
            except ImportError:
                import numpy as nx
                self.nx = nx
                self.backend = "NumPy"
        elif self.backend == "mlx":
            import mlx.core as nx
            self.nx = nx
            self.backend = "MLX"
        else:
            import numpy as nx
            self.nx = nx
            self.backend = "NumPy"
        
        self.e = self.nx.e
        self.float16 = self.nx.float16
        self.float32 = self.nx.float32
        self.int64 = self.nx.int64
        self.int32 = self.nx.int32
        self.bool = self.nx.bool_
    
    def array(self, x, dtype=None):
        if dtype is None:
            dtype = self.float32
        return self.nx.array(x, dtype=dtype)
    
    def sum(self, a, axis:Any=None, keepdims:bool=False, dtype=None):
        if dtype is None:
            dtype = self.nx.float32
        if self.backend == "MLX":
            if dtype is not None:
                a = self.nx.array(a, dtype=dtype)
            return self.nx.sum(a, axis=axis, keepdims=keepdims)
        
        return self.nx.sum(a,axis=axis,keepdims=keepdims,dtype=dtype)
    
    def float_32(self,x:Any) -> Any:
        if self.backend == "NumPy":
            return self.nx.float32(x)
        return self.nx.array(x, dtype=self.float32)
    
    def add_at(self, a:Any, idx, values) -> Any:
        if self.backend == "MLX":
            return a.at[idx].add(values)
        self.nx.add.at(a, idx, values)
        return a
    
    def zeros_like(self, a:Any, dtype=None):
        if dtype is None:
            dtype = self.nx.float32
        if self.backend == "MLX":
            a = self.nx.array(a, dtype=dtype)
        return self.nx.zeros_like(a)
    
    def zeros(self, size:Any, dtype=None) -> Any:
        if dtype is None:
            dtype = self.float32
        return self.nx.zeros(size, dtype=dtype)
    
    def ones_like(self, a:Any, dtype=None):
        if dtype is None:
            dtype = self.nx.float32
        if self.backend == "MLX":
            a = self.nx.array(a)
        return self.nx.ones_like(a, dtype=dtype)
    
    def ones(self, size:Any, dtype=None) -> Any:
        if dtype is None:
            dtype = self.nx.float32
        return self.nx.ones(size, dtype=dtype)
    
    def where(self, condition:Any,x:Any,y:Any) -> Any:
        return self.nx.where(condition,x,y)
    
    def triu(self, x, k=0, dtype=None):
        if dtype is None:
            dtype = self.nx.float32
        if self.backend == "MLX":
            x = self.nx.array(x, dtype=dtype)
            # if x.ndim == 1:
            #     n = x.shape[0]
            #     y = self.nx.zeros((n, n), dtype=x.dtype)
            #     for i in range(n):
            #         y[i, i:] = x[i:]
            #     return y
        return self.nx.triu(x, k=k)
    
    def max(self, x, axis=None, keepdims:bool=False):
        if self.backend == "MLX":
            x = self.nx.array(x)
        return self.nx.max(x,axis=axis, keepdims=keepdims)
    
    def min(self, x, axis=None, keepdims:bool=False):
        if self.backend == "MLX":
            x = self.nx.array(x)
        return self.nx.min(x,axis=axis, keepdims=keepdims)
    
    def exp(self, x, out=None, dtype=None):
        if dtype is None: 
            dtype = self.nx.float32
        if self.backend == "MLX":
            x = self.nx.array(x, dtype=dtype)
        if self.backend == "NumPy":
            return self.nx.exp(x, out=out, dtype=dtype)
        if out is not None:
            raise NotImplementedError("out is a no for mlx")
        return self.nx.exp(x)
    
    def clip(self, a:Any, a_min:Any, a_max:Any, dtype:Any=None) -> Any:
        if dtype is None: 
            dtype = self.nx.float32
        if self.backend == "MLX":
            a = self.nx.array(a, dtype=dtype)
            a_min = self.nx.array(a_min, dtype=dtype)
            a_max = self.nx.array(a_max, dtype=dtype)
            return self.nx.clip(a,a_min, a_max)
        return self.nx.clip(a,a_min, a_max,dtype=dtype)

    def log(self, a:Any, dtype:Any=None) -> Any:
        if dtype == None:
            dtype = self.nx.float32
        if self.backend == "MLX":
            a = self.nx.array(a, dtype)
            return self.nx.log(a)
        return self.nx.log(a, dtype=dtype)
    
    def arange(self, x:Any, y:Any=None, z:Any=None, dtype=None) -> Any:
        if dtype is None:
            dtype = self.nx.float32
        if self.backend == "MLX":
            return self.nx.array(self.nx.arange(x,y,z), dtype=dtype)
        return self.nx.arange(x,y,z, dtype=dtype)

    
    def indices(self, x:Any) -> Any:
        if self.backend == "NumPy":
            return self.nx.indices(x)
        ndims = len(x)
        result = []
        for axis, size in enumerate(x):
            idx = self.nx.arange(size)
            new_shape = [1] * ndims
            new_shape[axis] = size
            idx = idx.reshape(new_shape)
            idx = self.nx.broadcast_to(idx, x)
            result.append(idx)
        return self.nx.stack(result)
    
    def sqrt(self,x, dtype=None):
        if dtype is None:
            dtype = self.nx.float32
        if self.backend == "MLX":
            x = self.nx.array(x, dtype=dtype)
            return self.nx.sqrt(x)
        return self.nx.sqrt(x,dtype=dtype)   

    def uniform(self, low:float=0, high:float=1, size=None,*,dtype=None):
        if dtype is None:
            dtype = self.nx.float32
        if self.backend == "NumPy":
            if size is None:
                return dtype(self.nx.random.default_rng().uniform(low,high))
            else:
                return self.nx.random.default_rng().uniform(low,high, size=size).astype(dtype)

        if size is None:
            size = ()
        return self.nx.random.uniform(low, high, shape=size, dtype=dtype)
    
    def sliding_window_view(self, x:Any, window_shape:int, axis=None):
        if self.backend == "NumPy":
            return self.nx.lib.stride_tricks.sliding_window_view(x, window_shape, axis=axis)
        x = self.nx.array(x)   
        n = len(x)
        shape = (n - window_shape + 1,window_shape)
        strides = (1,1)
        return self.nx.as_strided(x,shape, strides)
    
    def mean(self,x, *,axis=None, keepdims:bool=False,dtype=None):
        if dtype is None:
            dtype = self.nx.float32

        if self.backend == "MLX":
            x = self.nx.array(x, dtype=dtype)
            return self.nx.mean(x, axis=axis,keepdims=keepdims)
        return self.nx.mean(x, axis=axis,keepdims=keepdims, dtype=dtype)
    
    def var(self, x,*,axis=None,keepdims=False,dtype=None):
        if dtype is None:
            dtype = self.nx.float32

        if self.backend == "MLX":
            x = self.nx.array(x, dtype=dtype)
            return self.nx.var(x, axis=axis,keepdims=keepdims)
        return self.nx.var(x, axis=axis,keepdims=keepdims, dtype=dtype)
    
    def dot(self,u,v):
        return u @ v
    
    def sin(self,a:Any,dtype=None) -> Any:
        if dtype is None:
            dtype = self.nx.float32
        if self.backend =="MLX":
            return self.nx.sin(self.nx.array(a, dtype=dtype))
        return self.nx.sin(a, dtype=dtype)
        
    def cos(self,a:Any,dtype=None) -> Any:
        if dtype is None:
            dtype = self.nx.float32
        if self.backend =="MLX":
            return self.nx.cos(self.nx.array(a, dtype=dtype))
        return self.nx.cos(a, dtype=dtype)
    
    def power(self,a,b, dtype=None) -> Any:
        if dtype is None:
            dtype = self.nx.float32
        if self.backend =="MLX":
            return self.nx.power(self.nx.array(a, dtype=dtype), self.nx.array(b,dtype=dtype))
        return self.nx.power(a,b,dtype=dtype)
    
    def argpartition(self,x, kth, axis=None):
        return self.nx.argpartition(x,kth, axis=axis)
    
    def random_choice(self,a, *, p=None):
        if self.backend == "MLX":
            cdf = self.nx.cumsum(p)
            r = self.nx.random.uniform()
            idx = self.nx.argmax(cdf >= r)
            return a[idx]
        return self.nx.random.choice(a, p=p)
    
    def copy(self,x):
        if self.backend == "MLX":
            return self.nx.array(x)
        
        return x.copy()
    
    def eval(self, *args):
        if self.backend == "MLX":
            self.nx.eval(*args)
        pass

    def concatenate(self,a):
        if self.backend == "MLX":
            return self.nx.concatenate(a)
        return self.nx.concatenate(a)
    
    def cumsum(self,a, *,axis=None, dtype=None):
        if dtype is None:
            dtype = self.nx.float32
        if self.backend == "MLX":
            a = self.nx.array(a, dtype=dtype)
            return self.nx.cumsum(a, axis=axis)
        return self.nx.cumsum(a, axis=axis, dtype=dtype)
    
    def argsort(self, a, axis=None):
        if self.backend == "MLX":
            a = self.nx.array(a)
        return self.nx.argsort(a, axis=axis)
    
    def all(self, a, *,axis=None, keepdims:bool=False):
        if self.backend == "MLX":
            a = self.nx.array(a)
        return self.nx.all(a, axis=axis, keepdims=keepdims)
    
    def argmax(self, a, axis=None, keepdims:bool=False):
        if self.backend == "MLX":
            a = self.nx.array(a)
        return self.nx.argmax(a, axis=axis, keepdims=keepdims)
    
    def abs(self, a):
        return self.nx.abs(a)
    
    def stack(self,a, axis=0):
        return self.nx.stack(a, axis=axis)
nx = Backend()
