from typing import Any
nx: Any
class Backend:
    def __init__(self, backend:str="auto") -> None:
        self.backend = backend.lower()
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
        
        print(f"using {self.backend}")

        self.float32 = self.nx.float32
        self.int64 = self.nx.int64
    
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
        return self.nx.zeros_like(self.nx.array(a, dtype=dtype))
    
    def zeros(self, size:Any, dtype=None) -> Any:
        if dtype is None:
            dtype = self.float32
        return nx.zeros(size, dtype = dtype)
    
    def ones_like(self, a:Any, dtype=None):
        if dtype is None:
            dtype = self.nx.float32
        return self.nx.ones_like(self.nx.array(a), dtype=dtype)
    
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
            if x.ndim == 1:
                n = x.shape[0]
                y = self.nx.zeros((n, n), dtype=x.dtype)
                for i in range(n):
                    y[i, i:] = x[i:]
                return y
        return self.nx.triu(x, k=k)
    
    def max(self, x, axis=None, keepdims:bool=False):
        return self.nx.max(self.nx.array(x),axis=axis, keepdims=keepdims)
    
    def exp(self, x, out=None, dtype=None):
        x = self.nx.array(x, dtype=dtype)
        if dtype is None: 
            dtype = self.nx.float32
        if self.backend == "NumPy":
            return self.nx.exp(x, out=out, dtype=dtype)
        if out is not None:
            raise NotImplementedError("out is a no for mlx")
       
        return self.nx.exp(x)
    

