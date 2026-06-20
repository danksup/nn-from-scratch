from engine.layer import Layer
from engine.attention import AttentionLayer
import numpy as np

class TransformerBlock:
    def __init__(self,embed_dim,ff_dim) -> None:
        self.attention = AttentionLayer(embed_dim)
        self.ff1 = Layer.hidden(ff_dim, embed_dim)
        self.ff2 = Layer.hidden(embed_dim, ff_dim)

    def forward(self,x):
        self.attn_out = self.attention.forward(x) + x
        self.ff_out = self.ff2.forward(self.ff1.forward(self.attn_out)) + self.attn_out
        return self.ff_out

    def backward(self, gradient):

        self.d_ff2 = self.ff2.backward(gradient)
        self.d_ff1 = self.ff1.backward(self.d_ff2)
        self.d_attn_out = self.d_ff1 + gradient
        self.d_attn = self.attention.backward(self.d_attn_out)
        self.dx = self.d_attn + self.d_attn_out

        return self.dx
        