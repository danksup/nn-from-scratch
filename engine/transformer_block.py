from engine.feedforward import Layer
from engine.attention import AttentionLayer
import numpy as np

class TransformerBlock:
    def __init__(self,embed_dim=None,ff_dim=None) -> None:
        if embed_dim is not None and ff_dim is not None:
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
    
    def to_dict(self):
        return {
            "attention":self.attention.to_dict(),
            "ff1":self.ff1.to_dict(),
            "ff2":self.ff2.to_dict(),
        } 
    
    @classmethod
    def from_dict(cls,thing):
        transformer_block = cls()
        transformer_block.attention = AttentionLayer.from_dict(thing["attention"])
        transformer_block.ff1 = Layer.from_dict(thing["ff1"])
        transformer_block.ff2 =  Layer.from_dict(thing["ff2"])
        return transformer_block