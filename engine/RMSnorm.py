from engine.backend import nx
from typing import Any

class RMSNorm:
    def __init__(self, embed_dim:int, epsilon:float=1e-5) -> None:
        self.gamma = nx.ones((embed_dim,), dtype=nx.float32)
        self.epsilon = nx.float_32(epsilon)

    def forward(self, x:Any) -> Any:
        self.input = x
        rms = nx.sqrt(nx.mean(x * x, axis=-1, keepdims=True) + self.epsilon)
        self.rms = rms
        self.normalized = x / rms
        return self.gamma * self.normalized
    
    def backward(self, gradient: Any) -> Any:
        self.d_gamma = nx.sum(gradient * self.normalized, axis=(0, 1), dtype=nx.float32)
        dx_norm = gradient * self.gamma
        d = self.input.shape[-1]
        sum_term = nx.sum(dx_norm * self.normalized, axis=-1, keepdims=True)
        dx = (dx_norm - (self.normalized / d) * sum_term) / self.rms
        return dx
    
    def to_dict(self)-> dict:
        return {
            "gamma":self.gamma.tolist(),
        }
    
    @classmethod
    def from_dict(cls, thing:dict) -> "RMSNorm": 
        rmsnorm = cls(len(thing["gamma"]))
        rmsnorm.gamma = nx.array(thing["gamma"],dtype=nx.float32)

        return rmsnorm