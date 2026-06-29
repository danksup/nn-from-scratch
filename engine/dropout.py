import engine.backend as nx
from typing import Any

class Dropout:
    """
    turns off some neurons by p rate so every neuron can learn
    """
    @staticmethod
    def _forward(x, p, is_training):
        assert 0 <= p < 1
        if not is_training:
            return x, None
        s = nx.uniform(0.0, 1.0, x.shape)
        mask = nx.where(s > p, 1 ,0)
        output = x * mask
        output = output / (1.0 - p)
        return output, mask
    
    @staticmethod
    def _backward(gradient, mask,p):
        assert 0 <= p < 1
        d_out = gradient * mask  / (1.0 - p)
        return d_out
    

