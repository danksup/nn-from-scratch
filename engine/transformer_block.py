from engine.swiglu import SwiGLU
from engine.attention import AttentionLayer
from engine.rmsnorm import RMSNorm
from engine.dropout import Dropout
from engine.rope import precompute_freqs
from engine.backend import nx
from typing import Any

class TransformerBlock:
    def __init__(self,embed_dim=None,ff_dim=None, n_heads=None) -> None:
        if embed_dim is not None and ff_dim is not None and n_heads is not None:
            self.embed_dim = embed_dim
            self.hidden_width = ff_dim
            self.n_heads = n_heads
            assert embed_dim % n_heads == 0
            self.head_dim = embed_dim // n_heads

            assert self.head_dim % 2 == 0, "head dim !% 2"
            self.freqs = precompute_freqs(self.head_dim)

            self.attention = AttentionLayer(embed_dim, n_heads)
            self.ff = SwiGLU(ff_dim, embed_dim)
            self.rmsnorm1 = RMSNorm(embed_dim)
            self.rmsnorm2 = RMSNorm(embed_dim)

            self.mask = None

    
    @nx.compile
    @staticmethod
    def _forward(x, causal_mask:Any, embed_dim:int, n_heads:int, head_dim:int,freqs:Any, Wqkv:Any, Wo:Any, Wcombined:Any, hidden_width:int, Wout:Any, epsilon:float, gamma1:Any, gamma2:Any, p:float, is_training:bool) -> tuple[Any, Any, Any]:
        fp16_x = x.astype(nx.float16)

        rmsnorm1_out, caches_rmsnorm1 = RMSNorm._forward(x, gamma1,epsilon)

        attn_out, caches_attn = AttentionLayer._forward(rmsnorm1_out,causal_mask, embed_dim, n_heads, head_dim, freqs, Wqkv, Wo,)
        attn_out, mask1 = Dropout._forward(attn_out, p,is_training)
        attn_out = attn_out + fp16_x

        rmsnorm2_out, caches_rmsnorm2 = RMSNorm._forward(attn_out, gamma2,epsilon)

        ff_out,caches_ff = SwiGLU._forward(rmsnorm2_out, hidden_width, Wcombined, Wout)
        ff_out, mask2 =  Dropout._forward(ff_out, p,is_training)
        ff_out = ff_out + attn_out
        
        
        masks = (mask1, mask2)
        caches = (caches_attn, caches_ff, caches_rmsnorm1, caches_rmsnorm2)
        return ff_out, masks, caches
    
    def forward(self, x:Any, is_training:bool) -> tuple[Any, Any, Any]:
        B,T,_ = x.shape
        Wqkv = self.attention.Wqkv
        Wo = self.attention.Wo
        Wcombined = self.ff.Wcombined
        Wout = self.ff.Wout
        epsilon = self.rmsnorm1.epsilon
        gamma1 = self.rmsnorm1.gamma
        gamma2 = self.rmsnorm2.gamma
        if self.mask is None or self.mask.shape[0] != T:
            self.mask = nx.triu(nx.ones((T, T), dtype=nx.bool), k=1)
        ff_out ,masks, caches = self._forward(x, self.mask, self.embed_dim, self.n_heads, self.head_dim,self.freqs, Wqkv, Wo, Wcombined, self.hidden_width, Wout, epsilon, gamma1, gamma2, 0.1, is_training)

        return ff_out, masks, caches

    @nx.compile
    @staticmethod
    def _backward(gradient:Any, mask1:Any, mask2:Any, caches_attn:tuple[Any,...], caches_ff:tuple[Any,...], caches_rmsnorm1:tuple[Any,...], caches_rmsnorm2:tuple[Any,...], d_attn_params:tuple[Any,...], gamma1:Any, gamma2:Any, ff_params:tuple) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
        d_ff = Dropout._backward(gradient, mask2, 0.1)
        dx_ff, dWout, dWcombined = SwiGLU._backward(d_ff, caches_ff, ff_params)

        d_rmsn2,d_gamma2 = RMSNorm._backward(dx_ff, caches_rmsnorm2 ,gamma2)

        d_attn_out = gradient + d_rmsn2

        d_attn = Dropout._backward(d_attn_out, mask1, 0.1)
        d_attn, dWqkv, dWo = AttentionLayer._backward(d_attn, caches_attn, d_attn_params)

        d_rmsn1, d_gamma1 = RMSNorm._backward(d_attn,caches_rmsnorm1,gamma1)

        dx = d_rmsn1 + d_attn_out

        return dx, dWout, dWcombined, dWqkv,dWo, d_gamma1, d_gamma2

    def backward(self, gradient: tuple[Any,...], masks:tuple[Any,...], all_caches:tuple[Any,...]) -> Any:
        caches_attn, caches_ff, caches_rmsnorm1, caches_rmsnorm2 = all_caches
        mask1, mask2 = masks
        d_attn_params = (self.n_heads, self.head_dim, self.embed_dim, self.attention.Wo, self.freqs, self.attention.Wqkv)
        ff_params = (self.ff.Wout, self.ff.Wcombined)
        dx, dWout, dWcombined,dWqkv,dWo, d_gamma1, d_gamma2 = self._backward(gradient, mask1=mask1, mask2=mask2, 
                                                            caches_attn=caches_attn, caches_ff=caches_ff, caches_rmsnorm1=caches_rmsnorm1, caches_rmsnorm2=caches_rmsnorm2, 
                                                            d_attn_params=d_attn_params, gamma1=self.rmsnorm1.gamma, gamma2=self.rmsnorm2.gamma, ff_params=ff_params)
        
        self.ff.dWout = dWout
        self.ff.dWcombined = dWcombined
        
        self.attention.dWqkv=dWqkv
        self.attention.dWo = dWo

        self.rmsnorm1.d_gamma = d_gamma1
        self.rmsnorm2.d_gamma = d_gamma2
        return dx

    def to_dict(self) -> dict:
        return {
            "attention":self.attention.to_dict(),
            "ff":self.ff.to_dict(),
            "rmsnorm1":self.rmsnorm1.to_dict(),
            "rmsnorm2":self.rmsnorm2.to_dict(),
        } 
    
    @classmethod
    def from_dict(cls,thing:dict) -> "TransformerBlock":
        transformer_block = cls()
        transformer_block.attention = AttentionLayer.from_dict(thing["attention"])
        transformer_block.ff = SwiGLU.from_dict(thing["ff"])
        transformer_block.rmsnorm1 = RMSNorm.from_dict(thing["rmsnorm1"])
        transformer_block.rmsnorm2 = RMSNorm.from_dict(thing["rmsnorm2"])
        return transformer_block

  