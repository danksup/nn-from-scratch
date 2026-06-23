from engine.backend import Backend
from typing import Any
nx = Backend()

class LayerNorm:
    def __init__(self, embed_dim:int, epsilon:float=1e-5) -> None:
        self.gamma = nx.ones((embed_dim,), dtype=nx.float32)
        self.beta = nx.zeros((embed_dim,), dtype=nx.float32)
        self.epsilon = nx.float_32(epsilon)

    def forward(self, x:Any) -> Any:
        mean = nx.mean(x, axis=-1, keepdims=True, dtype=nx.float32)
        self.var = nx.var(x, axis=-1, keepdims=True, dtype=nx.float32)
        self.x_norm = (x-mean) / nx.sqrt(self.var + self.epsilon, dtype=nx.float32)
        self.output = self.gamma * self.x_norm + self.beta

        return self.output
    def backward(self,gradient):
        self.d_beta =  nx.sum(gradient, axis=(0,1), dtype=nx.float32)
        self.d_gamma = nx.sum(gradient*self.x_norm, axis=(0,1), dtype=nx.float32)
        self.dx_norm = gradient * self.gamma
        inv_std = nx.float_32(1.0) / nx.sqrt(self.var + self.epsilon, dtype=nx.float32)
        dx = inv_std * (self.dx_norm - nx.mean(self.dx_norm, axis=-1,keepdims=True, dtype=nx.float32) - self.x_norm * nx.mean(self.dx_norm * self.x_norm, axis=-1,keepdims=True, dtype=nx.float32))

        return dx
    
    def to_dict(self)-> dict:
        return {
            "gamma":self.gamma.tolist(),
            "beta":self.beta.tolist()
        }
    
    @classmethod
    def from_dict(cls, thing:dict) -> "LayerNorm": 
        layernorm = cls(len(thing["gamma"]))
        layernorm.gamma = nx.array(thing["gamma"],dtype=nx.float32)
        layernorm.beta = nx.array(thing["beta"], dtype=nx.float32)

        return layernorm