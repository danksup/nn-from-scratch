import engine.backend as nx
from typing import Any

class Dropout:
    """
    turns off some neurons by p rate so every neuron can learn
    """
    @staticmethod
    def _forward(x, p, is_training):
        assert 0 <= p < 1
  
        s = nx.uniform(0.0, 1.0, x.shape)
        mask = nx.where(s > p, 1 ,0)
        mask = nx.where(is_training, mask, 1.0)
        output = (x * mask) / (1.0 - p if is_training else 1.0)
        return output, mask
    
    @staticmethod
    def _backward(gradient, mask,p):
        assert 0 <= p < 1
        d_out = gradient * mask  / (1.0 - p)
        return d_out
    

