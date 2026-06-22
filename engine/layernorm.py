import numpy as np

class LayerNorm:
    def __init__(self, embed_dim:int, epsilon:float=1e-5) -> None:
        self.gamma = np.ones((embed_dim,), dtype=np.float32)
        self.beta = np.zeros((embed_dim,), dtype=np.float32)
        self.epsilon = np.float32(epsilon)

    def forward(self, x:np.ndarray):
        mean = np.mean(x, axis=-1, keepdims=True, dtype=np.float32)
        self.var = np.var(x, axis=-1, keepdims=True, dtype=np.float32)
        self.x_norm = (x-mean) / np.sqrt(self.var + self.epsilon, dtype=np.float32)
        self.output = self.gamma * self.x_norm + self.beta

        return self.output
    def backward(self,gradient):
        self.d_beta =  np.sum(gradient, axis=(0,1), dtype=np.float32)
        self.d_gamma = np.sum(gradient*self.x_norm, axis=(0,1), dtype=np.float32)
        self.dx_norm = gradient * self.gamma
        inv_std = np.float32(1.0) / np.sqrt(self.var + self.epsilon, dtype=np.float32)
        dx = inv_std * (self.dx_norm - np.mean(self.dx_norm, axis=-1,keepdims=True, dtype=np.float32) - self.x_norm * np.mean(self.dx_norm * self.x_norm, axis=-1,keepdims=True, dtype=np.float32))

        return dx
    
    def to_dict(self)-> dict:
        return {
            "gamma":self.gamma.tolist(),
            "beta":self.beta.tolist()
        }
    
    @classmethod
    def from_dict(cls, thing:dict) -> "LayerNorm": 
        layernorm = cls(len(thing["gamma"]))
        layernorm.gamma = np.array(thing["gamma"],dtype=np.float32)
        layernorm.beta = np.array(thing["beta"], dtype=np.float32)

        return layernorm