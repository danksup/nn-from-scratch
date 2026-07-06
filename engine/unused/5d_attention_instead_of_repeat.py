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
    Args:
        embed_dim: number of embedding dimension of a token
        n_heads: number of query heads
        n_kv_heads: number of unique kv heads.
            shared by n_heads // n_kv_heads query heads(n_rep)
        """
    def __init__(self,embed_dim:int, n_heads:int, n_kv_heads:int=-1) -> None:
        self.n_kv_heads = n_kv_heads

        if n_kv_heads < 0:
            n_kv_heads = n_heads
        
        self.n_kv_heads = n_kv_heads
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        assert embed_dim % n_heads == 0
        assert n_heads % n_kv_heads == 0, "cant have more kv heads than query heads."
        self.head_dim = embed_dim // n_heads

        self.scale = nx.float_32(nx.sqrt(self.head_dim))
        scale = nx.float_32(1/self.scale)
        self.n_rep = self.n_heads // self.n_kv_heads 

        self.Wqkv = nx.uniform(-scale, scale, (embed_dim + 2 * n_kv_heads * self.head_dim, embed_dim), dtype=nx.float16) 
        self.Wo = nx.uniform(-scale,scale, (embed_dim,embed_dim), dtype=nx.float16) #projection
        self.dWqkv = None
        self.dWo = None
    
    @staticmethod
    def _forward(fp16_x:nx.ArrayLike, causal_mask:nx.ArrayLike, embed_dim:int, n_kv_heads:int, n_heads:int, n_rep:int, head_dim:int, freqs:int, Wqkv:nx.ArrayLike, Wo:nx.ArrayLike) -> tuple[nx.ArrayLike, tuple[nx.ArrayLike,...]]:
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
        #fp_16_x shape = (B,T,D)
        #Wqkv shape = (D + 2 * n_kv_heads * H, D), .T -> (D, D + 2 * n_kv_heads * H)
        #combined shape = (B, T, D + 2 * n_kv_heads * H)
        combined = fp16_x @ Wqkv.T 
        scale = nx.sqrt(head_dim)

        Q = combined[..., :embed_dim] #shape: (B, T, D)
        K = combined[..., embed_dim: embed_dim + (n_kv_heads * head_dim)]  #shape: (B, T, n_kv_heads * H)
        V = combined[..., embed_dim + (n_kv_heads * head_dim):] #shape: (B, T, n_kv_heads * H)

        B, T, _ = fp16_x.shape
        Q = Q.reshape(B, T, n_heads, head_dim).transpose(0,2,1,3)
        K = K.reshape(B, T, n_kv_heads, head_dim).transpose(0,2,1,3)
        V = V.reshape(B, T, n_kv_heads, head_dim).transpose(0,2,1,3)
        #each QKV shape = (B, n_heads, T, H)

        Q = rope_forward(Q, freqs)
        K = rope_forward(K, freqs)

        Q = Q.astype(nx.float32)
        K = K.astype(nx.float32)

        Q_5d = Q.reshape(B, n_kv_heads, n_rep, T, head_dim)
        K_5d = K.reshape(B, n_kv_heads, 1,     T, head_dim)
        V_5d = V.reshape(B, n_kv_heads, 1,     T, head_dim)
        #reshape to 5d instead of using repeat cus more efficient
        
        # repeats_K = nx.repeat(K,n_rep, axis=1 )
        # repeats_V = nx.repeat(V,n_rep, axis=1 )

        scores = (Q_5d @ K_5d.transpose(0, 1, 2, 4, 3)) / scale
        #causal mask makes it decoder only. cant look into future contexts.
        scores = nx.where(causal_mask, -1e9, scores)
        weights = softmax(scores).astype(nx.float16)

        output = weights @ V_5d
        output = output.reshape(B, n_heads, T, head_dim)
        output_concat = output.transpose(0, 2, 1, 3).reshape(B, T, embed_dim)
        output_projected = output_concat @ Wo
        cache =  (fp16_x, Q, K, V, weights, output_concat)
        return output_projected, cache
    
    @staticmethod
    def _backward(gradient:nx.ArrayLike, caches:tuple[Any,...], attn_params: tuple[Any,...]) -> tuple[nx.ArrayLike,...]:
        fp16_x, Q, K, V, weights, output_concat = caches
        n_heads, head_dim, embed_dim, n_kv_heads, n_rep, Wo, freqs, Wqkv = attn_params

        scale = nx.sqrt(head_dim)
        B, T, _ = fp16_x.shape
        d_output_concat = gradient @ Wo.T

        d_output_5d = d_output_concat.reshape(B, T, n_kv_heads, n_rep, head_dim).transpose(0, 2, 3, 1, 4)
        V_5d = V.reshape(B, n_kv_heads, 1, T, head_dim)
        dweights = d_output_5d @ V_5d.transpose(0, 1, 2, 4, 3)

        d_V_5d = weights.transpose(0, 1, 2, 4, 3) @ d_output_5d
        dV = d_V_5d.sum(axis=2) 

        dscores = softmax_derivative(weights, dweights)
        dscores /= scale
        
        K_5d = K.reshape(B, n_kv_heads, 1, T, head_dim)
        Q_5d = Q.reshape(B, n_kv_heads, n_rep, T, head_dim)
        
        d_Q_5d = dscores @ K_5d
        dQ = d_Q_5d.reshape(B, n_heads, T, head_dim)
        
        d_K_5d = dscores.transpose(0, 1, 2, 4, 3) @ Q_5d
        dK = d_K_5d.sum(axis=2) 

        dQ = rope_inverse(dQ, freqs)
        dK = rope_inverse(dK, freqs)

        dQ = dQ.transpose(0, 2, 1, 3).reshape(B, T, embed_dim)
        dK = dK.transpose(0, 2, 1, 3).reshape(B, T, n_kv_heads * head_dim)
        dV = dV.transpose(0, 2, 1, 3).reshape(B, T, n_kv_heads * head_dim)

        dQKV = nx.concatenate([dQ, dK, dV], axis=-1)

        DQKV = dQKV.reshape(-1, embed_dim + 2 * (n_kv_heads * head_dim))

        X = fp16_x.astype(nx.float32).reshape(-1, embed_dim)
        dWqkv = DQKV.T @ X

        H = output_concat.reshape(-1, embed_dim)
        G = gradient.reshape(-1, embed_dim)

        dWo = H.T @ G
        dx = dQKV @ Wqkv

        return dx,dWqkv,dWo
        
    def inference_forward(self, x, max_cache_len, freqs, cached_k=None, cached_v=None, position = 0):
        scale = nx.float_32(nx.sqrt(self.head_dim))
        combined = x @ self.Wqkv.T
        B, T, _ = x.shape    

        K = combined[..., self.embed_dim: self.embed_dim + (self.n_kv_heads * self.head_dim)] 

        K = K.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0,2,1,3)
        K = rope_forward(K, freqs, position) 
        K = K.astype(nx.float32)
        if cached_k is not None :
            cached_k = nx.concatenate([cached_k, K], axis = 2)
        else:
            cached_k = K

        # print(max(abs(K - cached_k[:, :, -1:, :])))
       
        V = combined[..., self.embed_dim + (self.n_kv_heads * self.head_dim):]
        V = V.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0,2,1,3)
        
        if cached_v is not None:
            cached_v = nx.concatenate([cached_v, V], axis = 2)
        else:
            cached_v = V

        if cached_k.shape[2] > max_cache_len:
            cached_k = cached_k[:, :, -max_cache_len:, :]

            cached_v = cached_v[:, :, -max_cache_len:, :]


        Q = combined[..., :self.embed_dim]
        Q = Q.reshape(B, T, self.n_heads, self.head_dim).transpose(0,2,1,3)
        Q = rope_forward(Q, freqs, position)

        Q = Q.astype(nx.float32)

        repeats_cached_k = nx.repeat(cached_k, self.n_rep, axis=1 )
        repeats_cached_v = nx.repeat(cached_v, self.n_rep, axis=1 )

        scores = (Q @ repeats_cached_k.transpose(0,1,3,2)) / scale
        weights = softmax(scores)
        output = weights @ repeats_cached_v
        output_concat = output.transpose(0, 2, 1, 3).reshape(B, T, self.embed_dim)
        output_projected = output_concat @ self.Wo
        
        return output_projected, cached_k, cached_v
    
    def to_dict(self) -> dict:
        '''serialize into dict with weights turned into list'''
        return {
            "embed_dim":self.embed_dim,
            "n_heads":self.n_heads,
            "n_kv_heads":self.n_kv_heads,
            "Wqkv":self.Wqkv.tolist(),
            "Wo":self.Wo.tolist(),
        }
    

    @classmethod
    def from_dict(cls,thing) -> "AttentionLayer":
        """deserialize"""
        embed_dim = thing["embed_dim"]
        n_kv_heads = thing["n_kv_heads"]
        n_heads = thing["n_heads"]
        Wqkv = thing["Wqkv"]
        Wo = thing["Wo"]

        attention = cls(embed_dim,n_heads, n_kv_heads)
        attention.Wqkv = nx.array(Wqkv, dtype=nx.float16)
        attention.Wo = nx.array(Wo, dtype=nx.float16)

        return attention