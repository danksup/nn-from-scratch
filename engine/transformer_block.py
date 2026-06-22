from engine.feedforward import Layer
from engine.attention import AttentionLayer
from engine.layernorm import LayerNorm
import numpy as np

class TransformerBlock:
    def __init__(self,embed_dim=None,ff_dim=None) -> None:
        if embed_dim is not None and ff_dim is not None:
            self.attention = AttentionLayer(embed_dim)
            self.ff1 = Layer.hidden(ff_dim, embed_dim)
            self.ff2 = Layer.hidden(embed_dim, ff_dim)
            self.layernorm1 = LayerNorm(embed_dim)
            self.layernorm2 = LayerNorm(embed_dim)

    def forward(self,x):
        ln1_out = self.layernorm1.forward(x)
        self.attn_out = self.attention.forward(ln1_out) + x

        ln2_out = self.layernorm2.forward(self.attn_out)
        self.ff_out = self.ff2.forward(self.ff1.forward(ln2_out)) + self.attn_out
        return self.ff_out

    def backward(self, gradient):
        self.d_ff2 = self.ff2.backward(gradient)
        self.d_ff1 = self.ff1.backward(self.d_ff2)
        self.d_ln2 = self.layernorm2.backward(self.d_ff1)
        self.d_attn_out = gradient +  self.d_ln2

        self.d_ln1 = self.layernorm1.backward(self.d_attn_out) 
        self.d_attn = self.attention.backward(self.d_ln1)

        self.dx = self.d_attn + self.d_attn_out

        return self.dx
    
    def to_dict(self):
        return {
            "attention":self.attention.to_dict(),
            "ff1":self.ff1.to_dict(),
            "ff2":self.ff2.to_dict(),
            "layernorm1":self.layernorm1.to_dict(),
            "layernorm2":self.layernorm2.to_dict()
        } 
    
    @classmethod
    def from_dict(cls,thing):
        transformer_block = cls()
        transformer_block.attention = AttentionLayer.from_dict(thing["attention"])
        transformer_block.ff1 = Layer.from_dict(thing["ff1"])
        transformer_block.ff2 =  Layer.from_dict(thing["ff2"])
        transformer_block.layernorm1 = LayerNorm.from_dict(thing["layernorm1"])
        transformer_block.layernorm2 = LayerNorm.from_dict(thing["layernorm2"])
        return transformer_block