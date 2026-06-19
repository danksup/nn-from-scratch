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

        scale = 1/np.sqrt(self.m)
        for i in range(n_neuron):
            for j in range(m_weight):
                self.weights[i,j] = random.uniform(-scale, scale)
            self.biases[i] = random.uniform(-scale, scale)
    
    def __repr__(self) -> str:
        return f"Weights: {self.weights}\nBiases: {self.biases}"

    def __len__(self):
        return self.n
    
    @classmethod
    def hidden(cls, n, m, activation=unit.leaky_relu, activation_derivative= unit.leaky_relu_derivative):
        return cls(n,m,activation ,activation_derivative)
    
    @classmethod
    def output(cls, n, m):
        return cls(n,m, None, None)
    
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

    def backward(self, err_signal:np.ndarray) -> np.ndarray:
        '''
        compute gradients and return previous layer error signal
        '''
        if self.activation_derivative is not None:
            current_neuron_error = err_signal * self.activation_derivative(self.last_z)
        else:
            current_neuron_error = err_signal

        batch_size = self.last_input.shape[0] if self.last_input.ndim > 1 else 1

        d_weight = ( current_neuron_error.T @ self.last_input) / batch_size
        previous_error = self.weights.T @ current_neuron_error.T
        self.d_weight = d_weight
        self.d_bias = current_neuron_error.mean(axis=0)

        
        return previous_error.T
        
