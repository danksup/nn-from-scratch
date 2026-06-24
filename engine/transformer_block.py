from engine.feedforward import Layer
from engine.attention import AttentionLayer
from engine.RMSnorm import RMSNorm

class TransformerBlock:
    def __init__(self,embed_dim=None,ff_dim=None) -> None:
        if embed_dim is not None and ff_dim is not None:
            self.attention = AttentionLayer(embed_dim)
            self.ff1 = Layer.hidden(ff_dim, embed_dim)
            self.ff2 = Layer.hidden(embed_dim, ff_dim)
            self.rmsnorm1 = RMSNorm(embed_dim)
            self.rmsnorm2 = RMSNorm(embed_dim)

    def forward(self,x):
        rmsn1_out = self.rmsnorm1.forward(x)
        self.attn_out = self.attention.forward(rmsn1_out) + x

        rmsn2_out = self.rmsnorm2.forward(self.attn_out)
        self.ff_out = self.ff2.forward(self.ff1.forward(rmsn2_out)) + self.attn_out
        return self.ff_out

    def backward(self, gradient):
        self.d_ff2 = self.ff2.backward(gradient)
        self.d_ff1 = self.ff1.backward(self.d_ff2)
        self.d_rmsn2 = self.rmsnorm2.backward(self.d_ff1)
        self.d_attn_out = gradient +  self.d_rmsn2

        self.d_attn = self.attention.backward(self.d_attn_out)
        self.d_rmsn1 = self.rmsnorm1.backward(self.d_attn) 


        self.dx = self.d_rmsn1 + self.d_attn_out

        return self.dx
    
    def to_dict(self):
        return {
            "attention":self.attention.to_dict(),
            "ff1":self.ff1.to_dict(),
            "ff2":self.ff2.to_dict(),
            "rmsnorm1":self.rmsnorm1.to_dict(),
            "rmsnorm2":self.rmsnorm2.to_dict()
        } 
    
    @classmethod
    def from_dict(cls,thing):
        transformer_block = cls()
        transformer_block.attention = AttentionLayer.from_dict(thing["attention"])
        transformer_block.ff1 = Layer.from_dict(thing["ff1"])
        transformer_block.ff2 =  Layer.from_dict(thing["ff2"])
        transformer_block.rmsnorm1 = RMSNorm.from_dict(thing["rmsnorm1"])
        transformer_block.rmsnorm2 = RMSNorm.from_dict(thing["rmsnorm2"])
        return transformer_block