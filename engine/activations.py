from engine.backend import Backend
from typing import Any
nx =  Backend("auto")
#rectified linear unit

def leaky_relu(x:Any) -> Any:
    return nx.where(x > 0, x, nx.float_32(0.01) * x)  #irl, x is unlikely to be 0 so we can ignore it

def leaky_relu_derivative(x:Any) -> Any:
    return nx.where(x > 0, nx.float_32(1.0), nx.float_32(0.01))

def softmax(x:Any, axis:Any=-1) -> Any:
    x = nx.array(x,dtype=nx.float32)
    max_x = nx.max(x, axis=axis, keepdims=True)
    exp_x = nx.exp(x - max_x, dtype=nx.float32)
    return exp_x / nx.sum(exp_x, axis=axis, keepdims=True, dtype=nx.float32)

def softmax_derivative(s:Any, grad:Any) -> Any:
    return  s * (grad -nx.sum(grad * s, axis=-1, keepdims=True,dtype=nx.float32))