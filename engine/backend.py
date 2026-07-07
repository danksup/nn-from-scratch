# pyright: reportArgumentType=false
# pyright: reportCallIssue=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportFunctionMemberAccess=false

from typing import Any, Union
import os
import mlx.core as mx
import numpy as np
ArrayLike = Union[mx.array, np.ndarray]
backend = os.environ.get("BACKEND", "auto").lower()

if backend == "auto":
    try:
        import mlx.core as _nx
        backend = "MLX"
    except ImportError:
        import numpy as _nx
        backend = "NumPy"
elif backend == "mlx":
    import mlx.core as _nx
    backend = "MLX"
else:
    import numpy as _nx
    backend = "NumPy"

e = _nx.e
float16 = _nx.float16
float32 = _nx.float32
int64 = _nx.int64
int32 = _nx.int32
bool_ = _nx.bool_

def array(x:Any, dtype=None) -> ArrayLike:
    if dtype is None:
        dtype = float32
    return _nx.array(x, dtype=dtype)

def sum(a:ArrayLike, axis:Any=None, keepdims:bool=False, dtype=None) -> Any:
    if dtype is None:
        dtype = _nx.float32
    if backend == "MLX":
        if dtype is not None:
            a = _nx.array(a, dtype=dtype)
        return _nx.sum(a, axis=axis, keepdims=keepdims)
    
    return _nx.sum(a,axis=axis,keepdims=keepdims,dtype=dtype)

def float_32(x:list | ArrayLike | float) -> Any:
    if backend == "NumPy":
        return _nx.float32(x)
    return _nx.array(x, dtype=float32)

def add_at(a:Any, idx:Any, values:Any) -> Any:
    if backend == "MLX":
        return a.at[idx].add(values)
    _nx.add.at(a, idx, values)
    return a

def zeros_like( a:ArrayLike, dtype=None) -> ArrayLike:
    if dtype is None:
        dtype = _nx.float32
    if backend == "MLX":
        a = _nx.array(a, dtype=dtype)
    return _nx.zeros_like(a)

def zeros( size:Any, dtype=None) -> ArrayLike:
    if dtype is None:
        dtype = float32
    return _nx.zeros(size, dtype=dtype)

def ones_like( a:ArrayLike, dtype=None) -> ArrayLike:
    if dtype is None:
        dtype = _nx.float32
    if backend == "MLX":
        a = _nx.array(a)
    return _nx.ones_like(a, dtype=dtype)

def ones( size:Any, dtype=None) -> ArrayLike:
    if dtype is None:
        dtype = _nx.float32
    return _nx.ones(size, dtype=dtype)

def where( condition:Any,x:Any,y:Any) -> ArrayLike:
    return _nx.where(condition,x,y)

def triu( x:ArrayLike, k=0, dtype=None) -> ArrayLike:
    if dtype is None:
        dtype = _nx.float32
    if backend == "MLX":
        x = _nx.array(x, dtype=dtype)
    return _nx.triu(x, k=k)

def max( x:ArrayLike, axis=None, keepdims:bool=False) -> ArrayLike:
    if backend == "MLX":
        x = _nx.array(x)
    return _nx.max(x, axis=axis, keepdims=keepdims)

def min( x:ArrayLike, axis=None, keepdims:bool=False) -> ArrayLike:
    if backend == "MLX":
        x = _nx.array(x)
    return _nx.min(x,axis=axis, keepdims=keepdims)

def exp( x:ArrayLike, out=None, dtype=None) -> ArrayLike:
    if dtype is None: 
        dtype = _nx.float32
    if backend == "MLX":
        x = _nx.array(x, dtype=dtype)
    if backend == "NumPy":
        return _nx.exp(x, out=out, dtype=dtype)
    if out is not None:
        raise NotImplementedError("out is a no for mlx")
    return _nx.exp(x)

def clip( a:Any, a_min:Any, a_max:Any, dtype:Any=None) -> ArrayLike:
    if dtype is None: 
        dtype = _nx.float32
    if backend == "MLX":
        a = _nx.array(a, dtype=dtype)
        a_min = _nx.array(a_min, dtype=dtype)
        a_max = _nx.array(a_max, dtype=dtype)
        return _nx.clip(a,a_min, a_max)
    return _nx.clip(a,a_min, a_max,dtype=dtype)

def log( a:Any, dtype:Any=None) -> ArrayLike:
    if dtype == None:
        dtype = _nx.float32
    if backend == "MLX":
        a = _nx.array(a, dtype)
        return _nx.log(a)
    return _nx.log(a, dtype=dtype)

def arange( x:Any, y:Any=None, z:Any=None, dtype=None) -> ArrayLike:
    if dtype is None:
        dtype = _nx.float32
    if backend == "MLX":
        return _nx.array(_nx.arange(x,y,z), dtype=dtype)
    return _nx.arange(x,y,z, dtype=dtype)


def indices( x:Any) -> Any:
    if backend == "NumPy":
        return _nx.indices(x)
    ndims = len(x)
    result = []
    for axis, size in enumerate(x):
        idx = _nx.arange(size)
        new_shape = [1] * ndims
        new_shape[axis] = size
        idx = idx.reshape(new_shape)
        idx = _nx.broadcast_to(idx, x)
        result.append(idx)
    return _nx.stack(result)

def sqrt(x:Any, dtype=None) -> Any:
    if dtype is None:
        dtype = _nx.float32
    if backend == "MLX":
        x = _nx.array(x, dtype=dtype)
        return _nx.sqrt(x)
    return _nx.sqrt(x,dtype=dtype)   

def uniform( low:float=0, high:float=1, size=None,*,dtype=None) -> Any:
    if dtype is None:
        dtype = _nx.float32
    if backend == "NumPy":
        if size is None:
            return dtype(_nx.random.default_rng().uniform(low,high))
        else:
            return _nx.random.default_rng().uniform(low,high, size=size).astype(dtype)

    if size is None:
        size = ()
    return _nx.random.uniform(low, high, shape=size, dtype=dtype)

def sliding_window_view( x:Any, window_shape:int, axis=None) -> ArrayLike:
    if backend == "NumPy":
        return _nx.lib.stride_tricks.sliding_window_view(x, window_shape, axis=axis)
    x = _nx.array(x)   
    n = len(x)
    shape = (n - window_shape + 1,window_shape)
    strides = (1,1)
    return _nx.as_strided(x,shape, strides)

def mean(x:ArrayLike, *,axis=None, keepdims:bool=False,dtype=None) -> ArrayLike:
    if dtype is None:
        dtype = _nx.float32

    if backend == "MLX":
        x = _nx.array(x, dtype=dtype)
        return _nx.mean(x, axis=axis,keepdims=keepdims)
    return _nx.mean(x, axis=axis,keepdims=keepdims, dtype=dtype)

def var( x:ArrayLike,*,axis=None,keepdims=False,dtype=None) -> ArrayLike:
    if dtype is None:
        dtype = _nx.float32

    if backend == "MLX":
        x = _nx.array(x, dtype=dtype)
        return _nx.var(x, axis=axis,keepdims=keepdims)
    return _nx.var(x, axis=axis,keepdims=keepdims, dtype=dtype)

def dot(u:ArrayLike,v:ArrayLike) -> Any:
    return u @ v

def sin(a:Any,dtype=None) -> Any:
    if dtype is None:
        dtype = _nx.float32
    if backend =="MLX":
        return _nx.sin(_nx.array(a, dtype=dtype))
    return _nx.sin(a, dtype=dtype)
    
def cos(a:Any,dtype=None) -> Any:
    if dtype is None:
        dtype = _nx.float32
    if backend =="MLX":
        return _nx.cos(_nx.array(a, dtype=dtype))
    return _nx.cos(a, dtype=dtype)

def power(a,b, dtype=None) -> Any:
    if dtype is None:
        dtype = _nx.float32
    if backend =="MLX":
        return _nx.power(_nx.array(a, dtype=dtype), _nx.array(b,dtype=dtype))
    return _nx.power(a,b,dtype=dtype)

def argpartition(x:ArrayLike, kth, axis=None) -> ArrayLike:
    return _nx.argpartition(x,kth, axis=axis)

def random_choice(a:ArrayLike, *, p=None) -> Any:
    if backend == "MLX":
        cdf = _nx.cumsum(p)
        r = _nx.random.uniform()
        idx = _nx.argmax(cdf >= r)
        return a[idx]
    return _nx.random.choice(a, p=p)

def copy(x:Any) -> Any:
    if backend == "MLX":
        return _nx.array(x)
    return x.copy()

def eval( *args):
    if backend == "MLX":
        _nx.eval(*args)
    pass

def concatenate(a:list[ArrayLike], axis=None) -> ArrayLike:
    return _nx.concatenate(a,axis=axis)

def cumsum(a:ArrayLike, *,axis=None, dtype=None) -> ArrayLike:
    if dtype is None:
        dtype = _nx.float32
    if backend == "MLX":
        a = _nx.array(a, dtype=dtype)
        return _nx.cumsum(a, axis=axis)
    return _nx.cumsum(a, axis=axis, dtype=dtype)

def argsort( a:ArrayLike, axis=None) -> Any:
    if backend == "MLX":
        a = _nx.array(a)
    return _nx.argsort(a, axis=axis)

def all( a:ArrayLike, *,axis=None, keepdims:bool=False) -> ArrayLike:
    if backend == "MLX":
        a = _nx.array(a)
    return _nx.all(a, axis=axis, keepdims=keepdims)

def argmax( a:ArrayLike, axis=None, keepdims:bool=False) -> Any:
    if backend == "MLX":
        a = _nx.array(a)
    return _nx.argmax(a, axis=axis, keepdims=keepdims)

def abs( a:Any) -> Any:
    return _nx.abs(a)

def stack(a:Any, axis=0)-> Any:
    return _nx.stack(a, axis=axis)

def any( a, *,keepdims:bool=False,axis=None):
    return _nx.any(a, keepdims=keepdims, axis=axis)

def isnan( a):
    return _nx.isnan(a)

def compile( fn=None):
    if backend == "MLX":
        return _nx.compile(fn) if fn is not None else _nx.compile
    def no_op_decorator(f):
        return f
    return no_op_decorator(fn) if fn is not None else no_op_decorator

def clear_cache():
    if backend == "MLX":
        _nx.clear_cache()
    pass

def repeat(a, repeats:int, axis:int=None):
    return _nx.repeat(a,repeats, axis=axis)

def logsumexp(a:ArrayLike,*,axis=None,keepdims=False) -> ArrayLike:
    if backend == "NumPy":
        m = _nx.max(a, axis=axis, keepdims=keepdims)
        return m + _nx.log(_nx.sum(_nx.exp(a - m), axis=axis, keepdims=keepdims))
    return _nx.logsumexp(a, axis=axis, keepdims=keepdims)

def norm(a:ArrayLike, ord:Any=None, axis=None, keepdims:bool=False):
    return _nx.linalg.norm(a, ord, axis=axis, keepdims=keepdims)