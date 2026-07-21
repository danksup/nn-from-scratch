import engine.backend as nx
from typing import Any

class Dropout:
    @staticmethod
    def _forward(x, p, is_training):
        assert 0 <= p < 1

        if not is_training:
            return x, None

        s = nx.uniform(0.0, 1.0, x.shape, dtype=x.dtype)
        mask = s > p

        scale = nx.array(1.0 / (1.0 - p), dtype=x.dtype)
        output = x * mask.astype(x.dtype) * scale

        return output, mask

    @staticmethod
    def _backward(gradient, mask, p):
        assert 0 <= p < 1

        if mask is None:
            return gradient

        scale = nx.array(1.0 / (1.0 - p), dtype=gradient.dtype)
        return gradient * mask.astype(gradient.dtype) * scale
    

