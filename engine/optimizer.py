import numpy as np

class Adam:
    def __init__(self, lr=0.01, beta:float=0.9, beta2:float=0.999, epsilon:float=1e-8) -> None:
        self.memory = {}
        self.lr = lr
        self.beta1 = beta
        self.beta2 = beta2
        self.epilon = epsilon
    def step(self, params:np.ndarray, gradient:np.ndarray):
        param_id = id(params)
        if param_id not in self.memory:
            self.memory[param_id] = {
                "momentum_estimate":np.zeros_like(params),
                "velocity":np.zeros_like(params),
                "time_step":0
            }
        current = self.memory[param_id]
        momentum_estimate = current["momentum_estimate"]
        velocity = current["velocity"]
        momentum_estimate *= self.beta1
        momentum_estimate += (1 - self.beta1) * gradient
        velocity *= self.beta2
        velocity += (1-self.beta2) * gradient ** 2
        current["time_step"] +=1 
        time_step = current["time_step"]
        m_hat =  momentum_estimate / (1- self.beta1 ** time_step)
        v_hat =  velocity / (1- self.beta2 ** time_step)
        params -= self.lr * m_hat / (np.sqrt(v_hat) + self.epilon)

class AdamW:
    def __init__(self, lr=0.01, beta:float=0.9, beta2:float=0.999, epsilon:float=1e-8, weight_decay=0.01) -> None:
        self.memory = {}
        self.lr = lr
        self.beta1 = beta
        self.beta2 = beta2
        self.epilon = epsilon
        self.weight_decay = weight_decay
    def step(self, params:np.ndarray, gradient:np.ndarray):
        param_id = id(params)
        if param_id not in self.memory:
            self.memory[param_id] = {
                "momentum_estimate":np.zeros_like(params),
                "velocity":np.zeros_like(params),
                "time_step":0
            }
        current = self.memory[param_id]
        momentum_estimate = current["momentum_estimate"]
        velocity = current["velocity"]
        momentum_estimate *= self.beta1
        momentum_estimate += (1 - self.beta1) * gradient
        velocity *= self.beta2
        velocity += (1-self.beta2) * gradient ** 2
        current["time_step"] +=1 
        time_step = current["time_step"]
        m_hat =  momentum_estimate / (1- self.beta1 ** time_step)
        v_hat =  velocity / (1- self.beta2 ** time_step)
        params -= self.lr * self.weight_decay * params
        params -= self.lr * m_hat / (np.sqrt(v_hat) + self.epilon)

class SGD:
    def __init__(self, lr=0.05) -> None:
        self.lr = lr
    
    # def step(self, layer:Layer) -> None:
    #     layer.weights -= self.lr * layer.d_weight
    #     layer.biases -= self.lr * layer.d_bias
    def step(self, params:np.ndarray,gradient:np.ndarray) -> None:
        params -= self.lr * gradient

class Momentum:
    def __init__(self, lr:float=0.01, beta:float=0.9) -> None:
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
        