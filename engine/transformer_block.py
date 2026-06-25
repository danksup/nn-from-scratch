from engine.feedforward import Layer
from engine.attention import AttentionLayer
from engine.RMSnorm import RMSNorm
from engine.dropout import Dropout
from engine.backend import nx

class TransformerBlock:
    def __init__(self,embed_dim=None,ff_dim=None, n_heads=None) -> None:
        if embed_dim is not None and ff_dim is not None and n_heads is not None:
            self.attention = AttentionLayer(embed_dim, n_heads)
            self.ff1 = Layer.hidden(ff_dim, embed_dim)
            self.ff2 = Layer.hidden(embed_dim, ff_dim)
            self.rmsnorm1 = RMSNorm(embed_dim)
            self.rmsnorm2 = RMSNorm(embed_dim)
            self.dropout1 = Dropout(0.1)
            self.dropout2 = Dropout(0.1)

    def forward(self, x):
        # print("input", nx.max(nx.abs(x)))

        rmsn1_out = self.rmsnorm1.forward(x)
        # print("after rms1", nx.max(nx.abs(rmsn1_out)))

        attn_out = self.attention.forward(rmsn1_out)
        # print("attention", nx.max(nx.abs(attn_out)))

        attn_out = self.dropout1.forward(attn_out)
        self.attn_out = attn_out + x
        # print("after residual1", nx.max(nx.abs(self.attn_out)))

        rmsn2_out = self.rmsnorm2.forward(self.attn_out)
        # print("after rms2", nx.max(nx.abs(rmsn2_out)))

        ff_out = self.ff2.forward(self.ff1.forward(rmsn2_out))
        # print("ff", nx.max(nx.abs(ff_out)))

        ff1_out = self.ff1.forward(rmsn2_out)
        # print("ff1", nx.max(nx.abs(ff1_out)))

        ff2_out = self.ff2.forward(ff1_out)
        # print("ff2", nx.max(nx.abs(ff2_out)))

        ff_out = self.dropout2.forward(ff_out)
        self.ff_out = ff_out + self.attn_out
        # print("after residual2", nx.max(nx.abs(self.ff_out)))

        return self.ff_out

    def backward(self, gradient):

        #ff
        d_ff = self.dropout2.backward(gradient)
        self.d_ff2 = self.ff2.backward(d_ff)
        self.d_ff1 = self.ff1.backward(self.d_ff2)

        self.d_rmsn2 = self.rmsnorm2.backward(self.d_ff1)

        #ff_out residual
        self.d_attn_out = gradient + self.d_rmsn2

        #attn
        self.d_attn = self.dropout1.backward(self.d_attn_out)
        self.d_attn = self.attention.backward(self.d_attn)

        self.d_rmsn1 = self.rmsnorm1.backward(self.d_attn)

        #attn_out residual
        self.dx = self.d_rmsn1 + self.d_attn_out

        return self.dx
    
    def train(self):
        self.dropout1.train()
        self.dropout2.train()

    def eval(self):
        self.dropout1.eval()
        self.dropout2.eval()
    
    def to_dict(self):
        return {
            "attention":self.attention.to_dict(),
            "ff1":self.ff1.to_dict(),
            "ff2":self.ff2.to_dict(),
            "rmsnorm1":self.rmsnorm1.to_dict(),
            "rmsnorm2":self.rmsnorm2.to_dict(),
            "dropout1": self.dropout1.p,
            "dropout2": self.dropout2.p,
        } 
    
    @classmethod
    def from_dict(cls,thing):
        transformer_block = cls()
        transformer_block.attention = AttentionLayer.from_dict(thing["attention"])
        transformer_block.ff1 = Layer.from_dict(thing["ff1"])
        transformer_block.ff2 =  Layer.from_dict(thing["ff2"])
        transformer_block.rmsnorm1 = RMSNorm.from_dict(thing["rmsnorm1"])
        transformer_block.rmsnorm2 = RMSNorm.from_dict(thing["rmsnorm2"])
        transformer_block.dropout1 = Dropout(thing["dropout1"])
        transformer_block.dropout2 = Dropout(thing["dropout2"])
        return transformer_block