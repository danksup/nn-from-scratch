from engine.moe import MoE
from engine.attention import AttentionLayer
from engine.rmsnorm import RMSNorm
from engine.dropout import Dropout
from engine.rope import precompute_freqs
import engine.backend as nx
from typing import Any

class TransformerBlock:
    def __init__(self,embed_dim,ff_dim, n_heads, n_kv_heads, n_experts=1, cf=1.25, top_k =2) -> None:
        self.causal_mask = None
        self.embed_dim = embed_dim
        self.hidden_width = ff_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads 
        assert embed_dim % n_heads == 0
        self.head_dim = embed_dim // n_heads
        self.n_experts = n_experts
        self.cf = cf

        assert self.head_dim % 2 == 0, "head dim !% 2"
        self.freqs = precompute_freqs(self.head_dim, 16384)

        self.attention = AttentionLayer(embed_dim, n_heads, n_kv_heads)
        self.ff = MoE(cf, top_k, n_experts, embed_dim, self.hidden_width)
        self.rmsnorm1 = RMSNorm(embed_dim)
        self.rmsnorm2 = RMSNorm(embed_dim)

    @nx.compile
    @staticmethod
    def _forward(x, causal_mask:Any, embed_dim:int, n_heads:int, n_kv_heads, n_rep, head_dim:int, n_experts, cf, top_k:int,freqs:Any, Wqkv:Any, Wo:Any, Wcombined:Any,router, hidden_width:int, Wout:Any, epsilon:float, gamma1:Any, gamma2:Any, p:float, is_training:bool) -> tuple[Any, Any, Any, Any, Any]:
        '''
        flow:
            input = x shape(B,T,D) -> rmsnorm(x) = rmsnorm_out -> attention(rmsnorm_out) + residual = attn_out
            \n
            -> rmsnorm(attn_out) = rmsnorm_out -> swiglu(rmsnorm_out)  -> ff_out + resudial = ff_out shape(B,T,D)
        '''
        
        fp16_x = x.astype(nx.float16)

        rmsnorm1_out, caches_rmsnorm1 = RMSNorm._forward(x, gamma1,epsilon)

        attn_out, caches_attn = AttentionLayer._forward(rmsnorm1_out,causal_mask, embed_dim, n_kv_heads,n_heads, n_rep, head_dim, freqs, Wqkv, Wo,)
        drop_attn_out, mask1 = Dropout._forward(attn_out, p,is_training)
        attn_out = drop_attn_out + fp16_x

        rmsnorm2_out, caches_rmsnorm2 = RMSNorm._forward(attn_out, gamma2,epsilon)

        ff_out, caches_ff, router_loss, normalized_histogram = MoE.forward(rmsnorm2_out, cf, top_k, router,n_experts,hidden_width,Wcombined, Wout)
        drop_ff_out, mask2 =  Dropout._forward(ff_out, p,is_training)
        ff_out = drop_ff_out + attn_out
        
        masks = (mask1, mask2)
        caches = (caches_attn, caches_ff, caches_rmsnorm1, caches_rmsnorm2)
        return ff_out, masks, caches, router_loss, normalized_histogram

    @nx.compile
    @staticmethod
    def _backward(gradient:Any, mask1:Any, mask2:Any, caches_attn:tuple[Any,...], caches_ff:tuple[Any,...], caches_rmsnorm1:tuple[Any,...], caches_rmsnorm2:tuple[Any,...], d_attn_params:tuple[Any,...], gamma1:Any, gamma2:Any, ff_params:tuple, moe_configs) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
        d_ff_drop = Dropout._backward(gradient, mask2, 0.1)
        dx_ff,  dWcombined, dWout, d_router = MoE.backward(d_ff_drop, caches_ff, moe_configs, ff_params)

        d_rmsn2,d_gamma2 = RMSNorm._backward(dx_ff, caches_rmsnorm2 ,gamma2)

        d_attn_out = gradient + d_rmsn2

        d_attn_drop = Dropout._backward(d_attn_out, mask1, 0.1)
        d_attn, dWqkv, dWo = AttentionLayer._backward(d_attn_drop, caches_attn, d_attn_params)

        d_rmsn1, d_gamma1 = RMSNorm._backward(d_attn,caches_rmsnorm1,gamma1)

        dx = d_rmsn1 + d_attn_out

        return dx, dWout, dWcombined, d_router, dWqkv,dWo, d_gamma1, d_gamma2
    
    def inference_forward(self, x, max_cache_len, cached_k=None, cached_v=None,  position=0,):
        fp16_x = x.astype(nx.float16)

        rmsnorm1_out, _ = RMSNorm._forward(x, self.rmsnorm1.gamma, self.rmsnorm1.epsilon)

        attn_out, cached_k, cached_v = self.attention.inference_forward(rmsnorm1_out,max_cache_len, self.freqs, cached_k, cached_v, position)
        attn_out = attn_out + fp16_x

        rmsnorm2_out, _ = RMSNorm._forward(attn_out, self.rmsnorm2.gamma, self.rmsnorm2.epsilon)

        ff_out,_,_,_ = MoE.forward(rmsnorm2_out, self.ff.cf, self.ff.top_k,self.ff.router, self.ff.n_experts, self.ff.hidden_width, self.ff.Wcombined, self.ff.Wout)
        ff_out = ff_out + attn_out
        
        return ff_out, cached_k, cached_v

    def to_dict(self) -> dict:
        return {
            "configs": {
                "cf":self.cf,
                "n_experts" :self.n_experts,
                "n_heads": self.n_heads,
                "hidden_width":self.hidden_width,
                "embed_dim":self.embed_dim,
                "n_kv_heads": self.n_kv_heads
            },
            "attention":self.attention.to_dict(),
            "ff":self.ff.to_dict(),
            "rmsnorm1":self.rmsnorm1.to_dict(),
            "rmsnorm2":self.rmsnorm2.to_dict(),
            # "causal_mask":self.causal_mask.tolist() if self.causal_mask is not None else None
        } 
    
    @classmethod
    def from_dict(cls,thing:dict) -> "TransformerBlock":
        configs = thing["configs"]
        transformer_block = cls(configs["embed_dim"], configs["hidden_width"], configs["n_heads"], configs["n_kv_heads"], configs["n_experts"],configs["cf"])
        transformer_block.attention = AttentionLayer.from_dict(thing["attention"])
        transformer_block.ff = MoE.from_dict(thing["ff"])
        transformer_block.rmsnorm1 = RMSNorm.from_dict(thing["rmsnorm1"])
        transformer_block.rmsnorm2 = RMSNorm.from_dict(thing["rmsnorm2"])
        # transformer_block.causal_mask = nx.array(thing["causal_mask"], dtype=nx.bool_)
        return transformer_block

  