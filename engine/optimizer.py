import numpy as np

class Adam:
    def __init__(self) -> None:
            pass

class SGD:
    def __init__(self, lr=0.05) -> None:
        self.lr = lr
    
    # def step(self, layer:Layer) -> None:
    #     layer.weights -= self.lr * layer.d_weight
    #     layer.biases -= self.lr * layer.d_bias
    def step(self, params:np.ndarray,gradient:np.ndarray) -> None:
        params -= self.lr * gradient

class Momentum:
    def __init__(self, lr:float=0.05, beta:float=0.95) -> None:
        self.lr = lr
        self.beta = beta
        self.velocities = {}
    
    # def step(self, layer:Layer):
    #     if layer not in self.velocities:
    #         self.velocities[layer] = {
    #             "velocity_weights": np.zeros_like(layer.weights),
    #             "velocity_biases": np.zeros_like(layer.biases)
    #          }
    #     current = self.velocities[layer]
    #     current_velocity_weights = current["velocity_weights"] *  self.beta + layer.d_weight
    #     current_velocity_biases = current["velocity_biases"] * self.beta + layer.d_bias
    #     layer.weights -= self.lr * current_velocity_weights
    #     layer.biases -= self.lr * current_velocity_biases
    #     self.velocities[layer] = {
    #          "velocity_weights":current_velocity_weights,
    #          "velocity_biases": current_velocity_biases,
    #     }

    def step(self, params:np.ndarray, gradient:np.ndarray):
        if id(params) not in self.velocities:
            self.velocities[id(params)] = np.zeros_like(params)
        current_velocity_params = self.velocities[id(params)] *  self.beta + gradient
        params -= self.lr * current_velocity_params
        self.velocities[id(params)] = current_velocity_params
        