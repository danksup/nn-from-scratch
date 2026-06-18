import random
import engine.activations as unit
import numpy as np

class Layer:
    def __init__(self, n_neuron:int, m_weight:int, activation=unit.leaky_relu, activation_derivative = unit.leaky_relu_derivative) -> None:
        '''
        Args:
            n_neuron: number of neurons
            m_weight: number of weights per neuron
            activation: activation unit
            activation_derivative: activation unit derivative
        '''
        self.n = n_neuron
        self.m = m_weight
        self.weights = np.ndarray((n_neuron,m_weight))
        self.biases = np.ndarray((n_neuron))
        self.activation = activation
        self.activation_derivative = activation_derivative

        for i in range(n_neuron):
            for j in range(m_weight):
                self.weights[i,j] = random.uniform(-1,1)
            self.biases[i] = random.uniform(-1,1)
    
    def __repr__(self) -> str:
        return f"Weights: {self.weights}\nBiases: {self.biases}"

    
    def forward(self, inputs:np.ndarray) -> np.ndarray:
        '''
        compute layer output and some values needed for layer.backward()
        '''
        weights = self.weights
        biases = self.biases
        
        if inputs.ndim == 1:
            z = weights @ inputs + biases
        else:
            z = inputs @ weights.T + biases

        self.last_z = z
        res = self.activation(z) if self.activation is not None else z
        self.last_ouput = res
        self.last_input = inputs

        return res

    def backward(self, err_signal:np.ndarray, lr = 1e-3) -> np.ndarray:
        '''
        backprop and update weights and biases.
        returns error contribution (blame signal) of previous layer
        '''
        if self.activation_derivative is not None:
            current_neuron_error = err_signal * self.activation_derivative(self.last_z)
        else:
            current_neuron_error = err_signal

        batch_size = self.last_input.shape[0] if self.last_input.ndim > 1 else 1

        d_weight = ( current_neuron_error.T @ self.last_input) / batch_size
        previous_error = self.weights.T @ current_neuron_error.T
        self.weights -= lr * d_weight
        self.biases -= lr * current_neuron_error.mean(axis=0)

        
        return previous_error.T
        
