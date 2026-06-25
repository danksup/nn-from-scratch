from engine.backend import nx
from typing import Any


class AdamW:
    def __init__(self, lr=0.01, beta:float=0.9, beta2:float=0.999, epsilon:float=1e-8, weight_decay=0.01) -> None:
        self.memory = {}
        self.lr = nx.float_32(lr)
        self.beta1 = nx.float_32(beta)
        self.beta2 = nx.float_32(beta2)
        self.epilon = nx.float_32(epsilon)
        self.weight_decay = nx.float_32(weight_decay)

    def step(self, name_param_gradient:tuple):
        name,param,gradient = name_param_gradient
        # print("before (step)", nx.max(nx.abs(param)))
        param_id = name
        if param_id not in self.memory:
            self.memory[param_id] = {
                "momentum_estimate":nx.zeros_like(param,dtype=nx.float32),
                "velocity":nx.zeros_like(param, dtype=nx.float32),
                "time_step":0
            }
        one = nx.float_32(1.0)
        norm = nx.sqrt(nx.sum(gradient ** 2), dtype=nx.float32)
        if norm > one:
            gradient = gradient * (one / norm)
        current = self.memory[param_id]
        momentum_estimate = current["momentum_estimate"]
        velocity = current["velocity"]
        momentum_estimate = self.beta1 * momentum_estimate + (one - self.beta1) * gradient
        velocity = self.beta2 * velocity + (one - self.beta2) * gradient * gradient
        current["momentum_estimate"] = momentum_estimate
        current["velocity"] = velocity
        current["time_step"] += 1 
        time_step = current["time_step"]
        m_hat =  nx.float_32(momentum_estimate / (one- self.beta1 ** time_step))
        v_hat =  nx.float_32(velocity / (one- self.beta2 ** time_step))
        param = param - self.lr * self.weight_decay * param
        param = param - self.lr * m_hat / (nx.sqrt(v_hat, dtype=nx.float32) + self.epilon)

        # print("after(step)", nx.max(nx.abs(param)))
        # print(name)
        # print(param.shape)
        # print(gradient.shape)
        # print(param.dtype)
        # print(gradient.dtype)
        # print(nx.max(nx.abs(param)))
        # print(nx.max(nx.abs(gradient)))
        # print("after clip", nx.max(nx.abs(gradient)))

        return param
    
    def step_many(self, params):
        optimized = {}
        for name, param, gradient in params:
            optimized[name] = self.step((name, param, gradient))
        return optimized

    def to_dict(self) -> dict:
        return self.memory

    @classmethod
    def from_dict(cls, thing):
        adam = cls()
        adam.memory = thing
        return adam

# class SGD:
#     def __init__(self, lr=0.05) -> None:
#         self.lr = lr
    
#     # def step(self, layer:Layer) -> None:
#     #     layer.weights -= self.lr * layer.d_weight
#     #     layer.biases -= self.lr * layer.d_bias
#     def step(self, params:np.ndarray,gradient:np.ndarray) -> None:
#         params -= self.lr * gradient

# class Momentum:
#     def __init__(self, lr:float=0.01, beta:float=0.9) -> None:
#         self.lr = lr
#         self.beta = beta
#         self.velocities = {}
    
#     # def step(self, layer:Layer):
#     #     if layer not in self.velocities:
#     #         self.velocities[layer] = {
#     #             "velocity_weights": np.zeros_like(layer.weights),
#     #             "velocity_biases": np.zeros_like(layer.biases)
#     #          }
#     #     current = self.velocities[layer]
#     #     current_velocity_weights = current["velocity_weights"] *  self.beta + layer.d_weight
#     #     current_velocity_biases = current["velocity_biases"] * self.beta + layer.d_bias
#     #     layer.weights -= self.lr * current_velocity_weights
#     #     layer.biases -= self.lr * current_velocity_biases
#     #     self.velocities[layer] = {
#     #          "velocity_weights":current_velocity_weights,
#     #          "velocity_biases": current_velocity_biases,
#     #     }

#     def step(self, params:np.ndarray, gradient:np.ndarray):
#         if id(params) not in self.velocities:
#             self.velocities[id(params)] = np.zeros_like(params)
#         current_velocity_params = self.velocities[id(params)] *  self.beta + gradient
#         params -= self.lr * current_velocity_params
#         self.velocities[id(params)] = current_velocity_params
        