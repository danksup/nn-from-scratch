from engine.backend import nx
from engine.activations import softmax, softmax_derivative
from engine.rope import precompute_freqs,rope_forward, rope_inverse
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
       
        assert self.head_dim % 2 == 0, "head dim !% 2"
        self.freqs = precompute_freqs(self.head_dim)
    
    @nx.nx.compile
    @staticmethod
    def _forward(fp16_x, embed_dim, n_heads, head_dim,freqs, Wqkv,scale, Wo):
        combined = fp16_x @ Wqkv.T

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
        mask = nx.triu(nx.ones((T, T), dtype=nx.bool), k=1)
        scores = nx.where(mask, -1e9, scores)
        weights = softmax(scores)
        output = weights @ V
        output_concat = output.transpose(0, 2, 1, 3).reshape(B, T, embed_dim)
        output_projected = output_concat @ Wo
        cache =  (fp16_x,Q,K, V, weights, output_concat)
        return output_projected, cache
    
    def forward(self, x:Any) -> tuple[Any, tuple]:
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
        fp16_x = x.astype(nx.float16)
        output_projected, cache = self._forward(fp16_x, self.embed_dim, self.n_heads, self.head_dim, self.freqs, self.Wqkv, self.scale, self.Wo)
        return output_projected, cache
    
    @nx.nx.compile
    @staticmethod
    def _backward(gradient,weights,fp16_x ,B, T, n_heads, head_dim, embed_dim, Wo, freqs, scale, K, Q,V, output_concat, Wqkv):
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

        return dWqkv,dWo, dx
    
    def backward(self, output_gradient:Any, cache:tuple) -> Any:
        """
        backprop
        """
        fp16_x, Q, K, V, weights, output_concat = cache
        B, T, _ = fp16_x.shape
        
        dWqkv, dWo, dx = self._backward(output_gradient,weights,fp16_x ,B, T, self.n_heads, self.head_dim, self.embed_dim, self.Wo, self.freqs, self.scale, K, Q,V, output_concat, self.Wqkv)
        self.dWqkv = dWqkv
        self.dWo = dWo
        return dx
        
    def to_dict(self) -> dict:
        '''serialize into dict with weights turned into list'''
        return {
            "embed_dim":self.embed_dim,
            "n_heads":self.n_heads,
            "Wqkv":self.Wqkv.tolist(),
            "Wo":self.Wo.tolist(),
        }
    
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
        