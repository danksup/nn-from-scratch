import engine.backend as nx
from typing import Any

class AdamW:
    def __init__(self, lr=0.01, beta:float=0.9, beta2:float=0.999, epsilon:float=1e-8, weight_decay=0.01) -> None:
        # self.memory = {}
        self.state = {}
        self.state["t"] = 0
        self.lr = nx.float_32(lr)
        self.beta1 = nx.float_32(beta)
        self.beta2 = nx.float_32(beta2)
        self.epsilon = nx.float_32(epsilon)
        self.weight_decay = nx.float_32(weight_decay)
    
    def step_many(self, name_param_gradient:list) -> dict:
        self.state["t"] += 1
        group = {}
        for name,param,gradient in name_param_gradient:
            shape = param.shape
            if shape not in group:
                group[shape] = []
            group[shape].append((name,param,gradient))

        optimized = {}
        for shape, thing in group.items():
            names = [i[0] for i in thing]
            params = nx.stack([i[1] for i in thing])
            gradients = nx.stack([i[2] for i in thing])
            if shape not in self.state:
                self.state[shape] = {
                    "m": nx.zeros_like(params),
                    "v": nx.zeros_like(params),
                    # "t": nx.float_32(.0)

                }
            state_shape = self.state[shape]    
            m_v_t = (state_shape["m"], state_shape["v"], self.state["t"])
            new_params, m,v,t = self.__step(m_v_t,params,gradients,self.lr,  self.epsilon, self.beta1, self.beta2, self.weight_decay)
            # self.state[shape] = {"m":m, "v":v, "t":t}
            self.state[shape] = {"m":m, "v":v}
            for idx, name in enumerate(names):
                optimized[name] = new_params[idx]
        return optimized
    
    @nx.compile
    @staticmethod
    def __step(m_v_t, params:Any, grads:Any, lr:float, epsilon:float, beta1:float, beta2:float, weight_decay:float) -> Any: 
        m,v,t = m_v_t       
        norm = nx.sqrt(nx.sum(grads**2, axis=tuple(range(1, grads.ndim)), keepdims=True))
        grads = nx.where(norm > 1.0, grads * (1.0 / (norm + epsilon)), grads)
        m = beta1 * m + (1.0 - beta1) * grads
        v = beta2 * v + (1.0 - beta2) * (grads**2)
        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)
        params = params - lr * weight_decay * params
        params = params - lr * m_hat / (nx.sqrt(v_hat) + epsilon)
        return params, m, v, t
    
    def to_dict(self) -> dict:
        return self.state

    @classmethod
    def from_dict(cls, thing):
        adam = cls()
        adam.state = thing
        return adam

