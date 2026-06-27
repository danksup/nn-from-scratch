from engine.backend import nx
from typing import Any

class AdamW:
    def __init__(self, lr=0.01, beta:float=0.9, beta2:float=0.999, epsilon:float=1e-8, weight_decay=0.01) -> None:
        # self.memory = {}
        self.state = {}
        self.lr = nx.float_32(lr)
        self.beta1 = nx.float_32(beta)
        self.beta2 = nx.float_32(beta2)
        self.epsilon = nx.float_32(epsilon)
        self.weight_decay = nx.float_32(weight_decay)

    def eval_state(self):
        state_arrays = []
        for state_dict in self.state.values():
            state_arrays.append(state_dict["m"])
            state_arrays.append(state_dict["v"])
        
        if state_arrays:
            nx.eval(*state_arrays)
            
    def step_many(self, name_param_gradient:list) -> dict:
        group = {}
        for name,param,gradient in name_param_gradient:
            shape = param.shape
            if shape not in group:
                group[shape] = []
            group[shape].append((name,param,gradient))

        optimized = {}
        for shape, thing in group.items():
            # print(f"Stacking shape {shape} with {len(thing)} items: {[i[0] for i in thing]}")
            names = [i[0] for i in thing]
            params = nx.stack([i[1] for i in thing])
            gradients = nx.stack([i[2] for i in thing])
            
            new_params = self._step(shape,params,gradients)

            for idx, name in enumerate(names):
                optimized[name] = new_params[idx]
        return optimized
    
    def _step(self, shape, params:Any, grads:Any) -> Any:
        if shape not in self.state:
            self.state[shape] = {
                "m": nx.zeros_like(params),
                "v": nx.zeros_like(params),
                "t": 0
            }
        
        norm = nx.sqrt(nx.sum(grads**2, axis=tuple(range(1, grads.ndim)), keepdims=True))
        grads = nx.where(norm > 1.0, grads * (1.0 / (norm + self.epsilon)), grads)
        state = self.state[shape]
        state["m"] = self.beta1 * state["m"] + (1.0 - self.beta1) * grads
        state["v"] = self.beta2 * state["v"] + (1.0 - self.beta2) * (grads**2)
        state["t"] += 1
        m_hat = state["m"] / (1.0 - self.beta1 ** state["t"])
        v_hat = state["v"] / (1.0 - self.beta2 ** state["t"])
        params = params - self.lr * self.weight_decay * params
        params = params - self.lr * m_hat / (nx.sqrt(v_hat) + self.epsilon)
        return params
    
    def to_dict(self) -> dict:
        return self.state

    @classmethod
    def from_dict(cls, thing):
        adam = cls()
        adam.state = thing
        return adam

