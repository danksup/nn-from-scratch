import engine.backend as nx
from engine.activations import softmax, softmax_derivative
from engine.rope import rope_forward, rope_inverse
from typing import Any

class AttentionLayer:
    """
    Explanation:
        Q (query): what the current token is looking for\n
        K (key): describe what each token contains\n
        V (key): What information is offered by each token\n
        """
    def __init__(self,embed_dim:int, n_heads:int) -> None:
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        assert embed_dim % n_heads == 0
        self.head_dim = embed_dim // n_heads

        self.scale = nx.float_32(nx.sqrt(self.head_dim))
        scale = nx.float_32(1/self.scale)

        # self.Wq = nx.uniform(-scale, scale, (embed_dim,embed_dim), dtype=nx.float16)
        # self.Wk = nx.uniform(-scale, scale, (embed_dim,embed_dim),dtype=nx.float16) 
        # self.Wv = nx.uniform(-scale, scale, (embed_dim,embed_dim),dtype=nx.float16) 
        self.Wqkv = nx.uniform(-scale, scale, (embed_dim * 3, embed_dim), dtype=nx.float16)
        self.Wo = nx.uniform(-scale,scale, (embed_dim,embed_dim), dtype=nx.float16) #projection
        self.dWqkv = None
        self.dWo = None
    
    @staticmethod
    def _forward(fp16_x, causal_mask, embed_dim, n_heads, head_dim, freqs, Wqkv, Wo):
        """
        f(x) = softmax((Q @ K.T) / scale) * V \n
        rope(x): inject position
        Q K.T: calculate how close 2 contexts are (dot products. the smaller the less relevant it is, the bigger the more relevant it is)\n
        softmax(Q K.T): turns it into probability distribution\n
        divide by scale = sqrt(head_dim): scaling to prevent numbers getting too large\n
        times by v: use weights to retrieve information. v contains information carried by each tokens\n
        \n
        return output projection Wo: all information from multiple heads back into one embedding size of embed_dim
        """
        combined = fp16_x @ Wqkv.T
        scale = nx.float_32(nx.sqrt(head_dim))

        Q = combined[..., :embed_dim]
        K = combined[..., embed_dim:2*embed_dim]
        V = combined[..., 2*embed_dim:]

        B, T, _ = fp16_x.shape
        Q = Q.reshape(B, T, n_heads, head_dim).transpose(0,2,1,3)
        K = K.reshape(B, T, n_heads, head_dim).transpose(0,2,1,3)
        V = V.reshape(B, T, n_heads, head_dim).transpose(0,2,1,3)

        Q = rope_forward(Q, freqs)
        K = rope_forward(K, freqs)

        Q = Q.astype(nx.float32)
        K = K.astype(nx.float32)

        scores = (Q @ K.transpose(0,1,3,2)) / scale

        #causal mask makes it decoder only. cant look into future contexts.
        scores = nx.where(causal_mask, -1e9, scores)
        weights = softmax(scores)
        output = weights @ V
        output_concat = output.transpose(0, 2, 1, 3).reshape(B, T, embed_dim)
        output_projected = output_concat @ Wo
        cache =  (fp16_x,Q,K, V, weights, output_concat)
        return output_projected, cache
    
    @staticmethod
    def _backward(gradient, caches, attn_params):
        fp16_x, Q, K, V, weights, output_concat = caches
        n_heads, head_dim, embed_dim, Wo, freqs, Wqkv = attn_params

        scale = nx.float_32(nx.sqrt(head_dim))
        B, T, _ = fp16_x.shape
        d_output_concat = gradient @ Wo.T
        d_output = d_output_concat.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
        dweights = d_output @ V.transpose(0, 1, 3, 2)
        dV = weights.transpose(0, 1, 3, 2) @ d_output
        dscores = softmax_derivative(weights, dweights)
        dscores /= scale
        dQ = dscores @ K
        dK = dscores.transpose(0, 1, 3, 2) @ Q

        dQ = rope_inverse(dQ, freqs)
        dK = rope_inverse(dK, freqs)

        dQ = dQ.transpose(0, 2, 1, 3).reshape(B, T, embed_dim)
        dK = dK.transpose(0, 2, 1, 3).reshape(B, T, embed_dim)
        dV = dV.transpose(0, 2, 1, 3).reshape(B, T, embed_dim)

        dQKV = nx.concatenate([dQ, dK,dV], axis=-1)
        DQKV = dQKV.reshape(-1, embed_dim * 3)

        X = fp16_x.astype(nx.float32).reshape(-1, embed_dim)
        dWqkv = (DQKV.T @ X) / (B * T)

        H = output_concat.reshape(-1, embed_dim)
        G = gradient.reshape(-1, embed_dim)

        dWo = (H.T @ G) / (B * T)
        dx = dQKV @ Wqkv

        return dx,dWqkv,dWo
        
    def to_dict(self) -> dict:
        '''serialize into dict with weights turned into list'''
        return {
            "embed_dim":self.embed_dim,
            "n_heads":self.n_heads,
            "Wqkv":self.Wqkv.tolist(),
            "Wo":self.Wo.tolist(),
        }
    
    def inference_forward(self, x, freqs, cached_k=None, cached_v=None):
        scale = nx.float_32(nx.sqrt(self.head_dim))
        combined = x @ self.Wqkv.T
        B, T, _ = x.shape

        if cached_k is None:
            position = 0
        else:
            position = cached_k.shape[2]

        K = combined[..., self.embed_dim:2*self.embed_dim]
        x.shape
        self.Wqkv.shape
        combined.shape
        K.shape
        K = K.reshape(B, T, self.n_heads, self.head_dim).transpose(0,2,1,3)
        K = rope_forward(K, freqs, position) 

        if cached_k is not None :
            cached_k = nx.concatenate([cached_k, K], axis = 2)
        else:
            cached_k = K
       
        V = combined[..., 2*self.embed_dim:]
        V = V.reshape(B, T, self.n_heads, self.head_dim).transpose(0,2,1,3)
        
        if cached_v is not None:
            cached_v = nx.concatenate([cached_v, V], axis = 2)
        else:
            cached_v = V

        Q = combined[..., :self.embed_dim]
        Q = Q.reshape(B, T, self.n_heads, self.head_dim).transpose(0,2,1,3)
        Q = rope_forward(Q, freqs, position)

        scores = (Q @ cached_k.transpose(0,1,3,2)) / scale
        weights = softmax(scores)
        output = weights @ cached_v
        output_concat = output.transpose(0, 2, 1, 3).reshape(B, T, self.embed_dim)
        output_projected = output_concat @ self.Wo
        return output_projected, cached_k, cached_v

    @classmethod
    def from_dict(cls,thing) -> "AttentionLayer":
        """deserialize"""
        embed_dim = thing["embed_dim"]
        n_heads = thing["n_heads"]
        Wqkv = thing["Wqkv"]
        Wo = thing["Wo"]

        attention = cls(embed_dim,n_heads)
        attention.Wqkv = nx.array(Wqkv, dtype=nx.float16)
        attention.Wo = nx.array(Wo, dtype=nx.float16)

        return attention
        