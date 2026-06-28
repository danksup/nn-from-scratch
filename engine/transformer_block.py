from engine.swiglu import SwiGLU
from engine.attention import AttentionLayer
from engine.rmsnorm import RMSNorm
from engine.dropout import Dropout
from engine.backend import nx
from typing import Any

class TransformerBlock:
    def __init__(self,embed_dim=None,ff_dim=None, n_heads=None) -> None:
        if embed_dim is not None and ff_dim is not None and n_heads is not None:
            self.attention = AttentionLayer(embed_dim, n_heads)
            self.ff = SwiGLU(ff_dim, embed_dim)
            self.rmsnorm1 = RMSNorm(embed_dim)
            self.rmsnorm2 = RMSNorm(embed_dim)
            self.dropout1 = Dropout(0.1)
            self.dropout2 = Dropout(0.1)

    def forward(self, x:Any) -> Any:
        rmsn1_out, rmsn1_caches = self.rmsnorm1.forward(x)

        attn_out, attention_caches = self.attention.forward(rmsn1_out)
        attn_out = self.dropout1.forward(attn_out)
        attn_out = attn_out + x

        rmsn2_out, rmsn2_caches = self.rmsnorm2.forward(attn_out)

        ff_out,ff_caches = self.ff.forward(rmsn2_out)
        ff_out = self.dropout2.forward(ff_out)
        ff_out = ff_out + attn_out

        return ff_out, attention_caches, ff_caches, rmsn1_caches, rmsn2_caches

    def backward(self, gradient:Any, attention_caches:Any, ff_caches:Any, rmsn1_caches, rmsn2_caches) -> Any:
        #ff
        d_ff = self.dropout2.backward(gradient)
        dx_ff = self.ff.backward(d_ff, ff_caches)

        d_rmsn2 = self.rmsnorm2.backward(dx_ff, rmsn2_caches)

        #ff_out residual
        d_attn_out = gradient + d_rmsn2

        #attn
        d_attn = self.dropout1.backward(d_attn_out)
        d_attn = self.attention.backward(d_attn, attention_caches)

        d_rmsn1 = self.rmsnorm1.backward(d_attn, rmsn1_caches)

        #attn_out residual
        dx = d_rmsn1 + d_attn_out

        return dx
    
    def train(self) -> None:
        self.dropout1.train()
        self.dropout2.train()

    def eval(self) -> None:
        self.dropout1.eval()
        self.dropout2.eval()
    
    def to_dict(self) ->dict:
        return {
            "attention":self.attention.to_dict(),
            "ff":self.ff.to_dict(),
            "rmsnorm1":self.rmsnorm1.to_dict(),
            "rmsnorm2":self.rmsnorm2.to_dict(),
            "dropout1": self.dropout1.p,
            "dropout2": self.dropout2.p,
        } 
    
    @classmethod
    def from_dict(cls,thing:dict) -> "TransformerBlock":
        transformer_block = cls()
        transformer_block.attention = AttentionLayer.from_dict(thing["attention"])
        transformer_block.ff = SwiGLU.from_dict(thing["ff"])
        transformer_block.rmsnorm1 = RMSNorm.from_dict(thing["rmsnorm1"])
        transformer_block.rmsnorm2 = RMSNorm.from_dict(thing["rmsnorm2"])
        transformer_block.dropout1 = Dropout(thing["dropout1"])
        transformer_block.dropout2 = Dropout(thing["dropout2"])
        return transformer_block

  