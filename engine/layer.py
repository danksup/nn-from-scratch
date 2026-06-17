import random
import engine.activations as unit
from engine.neuron import forward

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
        self.weights = []
        self.biases = []
        self.activation = activation
        self.activation_derivative = activation_derivative

        for _ in range(n_neuron):
            weight_row = []
            for _ in range(m_weight):
                ran_weight = random.uniform(-1,1)
                weight_row.append(ran_weight)

            self.biases.append(random.uniform(-1,1))
            
            self.weights.append(weight_row)
    
    def __repr__(self) -> str:
        return f"Weights: {self.weights}\nBiases: {self.biases}"

    
    def forward(self, inputs:list):
        '''
        compute layer output and some values needed for layer.backward()
        '''
        res = []
        self.last_z = []
        for weights, bias in zip(self.weights, self.biases):
            self.last_z.append(forward(inputs, weights, bias, active=False))
            if self.activation is not None:
                res.append(forward(inputs, weights, bias, activation=self.activation))
            else:
                res.append(forward(inputs, weights, bias, active=False))

        self.last_input = inputs
        self.last_output = res

        return res

    def backward(self, err_signal:list, lr = 1e-3) -> list:
        '''
        backprop and update weights and biases.
        returns error contribution (blame signal) of previous layer
        '''
        previous_error = [0] * len(self.last_input)

        for i in range(self.n):
            if self.activation_derivative is not None:
                current_neuron_error = err_signal[i] * self.activation_derivative(self.last_z[i])
            else:
                current_neuron_error = err_signal[i] 

            for ix in range(len(self.weights[i])):
                previous_error[ix] += current_neuron_error * self.weights[i][ix]
                d_weight = current_neuron_error * self.last_input[ix]
                self.weights[i][ix] -= lr * d_weight
            
            self.biases[i] -= lr * current_neuron_error
        
        return previous_error
        
