from engine.backend import nx
from typing import Any
class Dropout:
    """
    turns off some neurons by p rate so every neuron can learn
    """
    def __init__(self,p) -> None:
        self.is_training:bool=True
        self.p = p
        assert 0 <= p < 1

    def forward(self, x) -> Any:
        """
        turn off some neurons
        """
        if not self.is_training:
            return x
        s = nx.uniform(0.0, 1.0, x.shape)
        self.mask = nx.where(s > self.p, 1 ,0)
        output = x * self.mask
        output = output / (1.0 - self.p)
        return output
    
    def backward(self, gradient):
        "backprop"
        d_out = gradient * self.mask
        d_out = d_out / (1.0 - self.p)
        return d_out
    
    def train(self):
        self.is_training = True

    def eval(self):
        self.is_training = False
