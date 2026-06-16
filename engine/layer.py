import random
import engine.activations as unit
from engine.neuron import forward

class Layer:
    def __init__(self, n_neuron:int, m_weight:int, activation=unit.leaky_relu) -> None:
        '''
        n_neuron = number of neurons
        m_weight = number of weights per neuron
        activation = activation unit
        '''
        self.n = n_neuron
        self.weights = []
        self.biases = []
        self.activation = activation

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
        res = []
        self.last_z = []
        for weights, bias in zip(self.weights, self.biases):
            self.last_z.append(forward(inputs, weights, bias, False))
            res.append(forward(inputs, weights, bias, activation=self.activation))

        self.last_input = inputs
        self.last_output = res

        return res

    def backward(self, err_signal:list, lr = 1e-3):
        for idx
        pass
