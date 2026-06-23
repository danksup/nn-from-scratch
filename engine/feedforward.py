import random
import engine.activations as unit
import numpy as np

class Layer:
    def __init__(self, n_neuron:int, m_weight:int, layer_type:str="hidden",activation=unit.leaky_relu, activation_derivative = unit.leaky_relu_derivative) -> None:
        '''
        Args:
            n_neuron: number of neurons
            m_weight: number of weights per neuron
            activation: activation unit
            activation_derivative: activation unit derivative
        '''
        self.layer_type = layer_type
        self.n = n_neuron
        self.m = m_weight
        self.activation = activation
        self.activation_derivative = activation_derivative
        scale = np.float32(1.0/np.sqrt(self.m))
        rng = np.random.default_rng()
        self.weights = rng.uniform(low=-scale, high=scale,size=(self.n, self.m)).astype(np.float32)
        self.biases = rng.uniform(low=-scale, high=scale,size=(self.n,)).astype(np.float32)
    
    def __repr__(self) -> str:
        return f"Weights: {self.weights}\nBiases: {self.biases}"

    def __len__(self):
        return self.n
    
    @classmethod
    def hidden(cls, n:int, m:int, activation=unit.leaky_relu, activation_derivative= unit.leaky_relu_derivative) -> "Layer":
        return cls(n,m,"hidden",activation ,activation_derivative)
    
    @classmethod
    def output(cls, n:int, m:int) -> "Layer":
        return cls(n,m, "output",None, None)
    
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

        batch_size, seq_len, _ = self.last_input.shape
        # X = self.last_input.reshape(-1, self.last_input.shape[-1])
        # E = current_neuron_error.reshape(-1, current_neuron_error.shape[-1])
        X = np.ascontiguousarray(self.last_input).reshape(-1, self.last_input.shape[-1]) #ai suggested this, because machine loves when the data are contigous so they can just take it without jumping (row since it's based on C)
        E = np.ascontiguousarray(current_neuron_error).reshape(-1, current_neuron_error.shape[-1])
        self.d_weight = (E.T @ X) / (batch_size * seq_len)
        # self.d_weight = np.einsum("bso,bsi->oi",current_neuron_error,self.last_input, dtype=np.float32) / np.float32(batch_size * seq_len)
        previous_error = current_neuron_error @ self.weights

        self.d_bias = current_neuron_error.mean(axis=(0,1),dtype=np.float32)

        return previous_error
    
    def to_dict(self) -> dict:
        return {
            "n":self.n,
            "m":self.m,
            "weights":self.weights.tolist(),
            "biases":self.biases.tolist(),
            "type":self.layer_type
        }
    
    @classmethod
    def from_dict(cls,thing:dict) -> "Layer":
        n = thing['n']
        m = thing['m']
        weights = np.array(thing["weights"], dtype=np.float32)
        biases = np.array(thing["biases"],np.float32)
        layer_type = thing["type"]
        if layer_type == "hidden":
            ff = cls.hidden(n,m)
        elif layer_type == "output":
            ff = cls.output(n,m)
        else:
            raise ValueError(f"unknown type {layer_type}")
        ff.weights = weights
        ff.biases = biases

        return ff
        
        
