import numpy as np

#rectified linear unit
def leaky_relu(x:np.ndarray):
    return np.where(x > 0, x, 0.01 * x)  #irl, x is unlikely to be 0 so we can ignore it

def leaky_relu_derivative(x:np.ndarray):
    return np.where(x > 0, 1, 0.01)

def softmax(x, axis=-1) :
    max_x = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def softmax_derivative(s, grad):
    return  s * (grad -np.sum(grad * s, axis=-1, keepdims=True))