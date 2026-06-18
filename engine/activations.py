import numpy as np

#rectified linear unit
def leaky_relu(x:np.ndarray):
    return np.where(x > 0, x, 0.01 * x)  #irl, x is unlikely to be 0 so we can ignore it

def leaky_relu_derivative(x:np.ndarray):
    return np.where(x > 0, 1, 0.01)

def softmax(l:np.ndarray) -> np.ndarray:
    if l.ndim == 2:
        max_l = np.max(l, axis=1).reshape(-1,1)
        exponent = np.exp(l - max_l) 
        return exponent/np.sum(exponent, axis=-1).reshape(-1,1) 
    else:
        max_l = np.max(l)
        exponent = np.exp(l - max_l)
        return exponent / np.sum(exponent)