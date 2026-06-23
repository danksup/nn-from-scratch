from .activations import leaky_relu, leaky_relu_derivative
from .utils import dot


#1 neuron
def forward(inputs:list, weights:list, b:float, active:bool=True, activation = leaky_relu) -> float:
    '''
    Args:
        inputs: list of inputs
        weights: list of weights
        b: bias

    output: 0 or 1
    '''
    return activation(dot(inputs, weights) + b) if active else (dot(inputs,weights) + b)
