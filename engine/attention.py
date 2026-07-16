import engine.backend as nx
from engine.activations import softmax, softmax_derivative
from engine.rope import rope_forward, rope_inverse
from typing import Any

class AttentionLayer:
    def __init__(self,embed_dim:int, n_heads:int, n_kv_heads:int=-1, W=8) -> None:
        self.n_kv_heads = n_kv_heads

        if n_kv_heads < 0:
            n_kv_heads = n_heads
        
        self.n_kv_heads = n_kv_heads
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        assert embed_dim % n_heads == 0
        assert n_heads % n_kv_heads == 0, "cant have more kv heads than query heads."
        self.head_dim = embed_dim // n_heads
        self.W = W

        self.scale = nx.float_32(nx.sqrt(self.head_dim))
        scale = nx.float_32(1/self.scale)
        self.n_rep = self.n_heads // self.n_kv_heads 

        self.Wqkv = nx.uniform(-scale, scale, (embed_dim + 2 * n_kv_heads * self.head_dim, embed_dim), dtype=nx.float16) 
        self.Wo = nx.uniform(-scale,scale, (embed_dim,embed_dim), dtype=nx.float16) #projection
        self.dWqkv = None
        self.dWo = None

    @staticmethod
    def _forward(fp16_x:nx.ArrayLike, causal_mask:nx.ArrayLike, embed_dim:int, n_kv_heads:int, n_heads:int, n_rep:int, head_dim:int, W, freqs:int, Wqkv:nx.ArrayLike, Wo:nx.ArrayLike):
        combined = fp16_x @ Wqkv.T 
        SCALE = nx.float_32(nx.sqrt(head_dim))

        Q = combined[..., :embed_dim] #shape: (B, T, D)
        K = combined[..., embed_dim: embed_dim + (n_kv_heads * head_dim)]  #shape: (B, T, n_kv_heads * H)
        V = combined[..., embed_dim + (n_kv_heads * head_dim):] #shape: (B, T, n_kv_heads * H)
        
        B, T, _ = fp16_x.shape
        Q = Q.reshape(B, T, n_heads, head_dim).transpose(0,2,1,3) #(B,n_heads,T, Dh)
        K = K.reshape(B, T, n_kv_heads, head_dim).transpose(0,2,1,3) #(B, n_kv_heads, T, Dh)
        V = V.reshape(B, T, n_kv_heads, head_dim).transpose(0,2,1,3) #(B, n_kv_heads, T, Dh)

        Q = rope_forward(Q, freqs)
        K = rope_forward(K, freqs)

        Q = Q.astype(nx.float32)
        K = K.astype(nx.float32)

        pad = [(0,0), (0,0),(W,0), (0,0)]
        P = T + W
        stride = (head_dim * n_kv_heads * P, P * head_dim, head_dim, head_dim, 1)
        padded_K = nx.pad(K, pad, constant_value=0)  #(B, n_kv_head, T+W, Dh)
        padded_V = nx.pad(V, pad, constant_value=0) #(B,n_kv_head,T+W, Dh)
        shape = (padded_K.shape[0], padded_K.shape[1], T, W + 1, head_dim) #(B, n_kv_head, T, W+1, Dh)
        windows_K = nx.as_strided(padded_K, shape=shape,strides=stride) #shape=shape
        windows_V = nx.as_strided(padded_V, shape=shape,strides=stride) #shape=shape

        repeat_K = nx.repeat(windows_K, n_rep, axis=1) #(B, n_kv_head * n_rep, T, W+1, Dh)
        repeat_V = nx.repeat(windows_V, n_rep, axis=1) #(B, n_kv_head * n_rep, T, W+1, Dh)

        Q_5d = Q[:,:,:,None,:] #(B,n_heads,T, 1, Dh)
        scores = (Q_5d @ repeat_K.transpose(0,1,2,4,3)) / SCALE #(B, n_heads, T, 1, W+1)
        scores = scores[:,:,:,0,:] #(B, n_heads, T, W+1)
        scores = nx.where(causal_mask, -1e9, scores)
        weights = softmax(scores) #(B, n_heads, T, W+1)
        output = weights[:,:,:,None,:] @ repeat_V #(B, n_heads, T, 1, Dh)
        output = output[:,:,:,0,:] #(B, n_heads, T, Dh)
        output_concat = output.transpose(0, 2, 1, 3).reshape(B, T, embed_dim)
        output_projected = output_concat @ Wo #B,T,D
        cache = (fp16_x, Q, repeat_K, repeat_V, weights, output_concat)
        return output_projected, cache
    
    @staticmethod
    def _backward(gradient:nx.ArrayLike, caches:tuple[Any,...], attn_params: tuple[Any,...]) -> tuple[nx.ArrayLike,...]:
        fp16_x, Q, repeat_K, repeat_V, weights, output_concat = caches
        n_heads, head_dim, embed_dim, n_kv_heads, n_rep, W, Wo, freqs, Wqkv = attn_params

        scale = nx.float_32(nx.sqrt(head_dim))
        B, T, D = fp16_x.shape
        d_output_concat = gradient @ Wo.T #B,T,D
        d_output = d_output_concat.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3) #(B, n_heads, T,  Dh)
        
        d_output_5d = d_output[:,:,:,None,:] #B, n_heads, T, 1, Dh
        d_weights =  d_output_5d @ repeat_V.transpose(0,1,2,4,3) #(B, n_heads, T, 1, W+1)
        d_weights = d_weights[:,:,:,0,:]#(B, n_heads, T, W+1)

                #(B, n_heads, T, W+1, 1)
        d_repeatV = weights[...,None] @ d_output_5d #(B, n_heads, T, W+1, Dh)
        d_repeatV = d_repeatV.reshape(B, n_kv_heads, n_rep, T , W+1, head_dim)
        d_windows_V = nx.sum(d_repeatV, axis=2) #(B, n_kv_heads, T , W+1, head_dim)

        d_scores = softmax_derivative(weights, d_weights) #(B, n_heads, T, W+1)
        d_scores /= scale

        d_scores_5d = d_scores[:,:,:,None,:] #(B, n_heads, T, 1, W+1)
        dQ = d_scores_5d @ repeat_K #(B, n_heads, T, 1, Dh)
        dQ = dQ[:,:,:,0,:] #(B, n_heads, T, Dh)
        Q_5d = Q[:,:,:,None,:] #(B,n_heads,T, 1, Dh)
        d_repeatK = d_scores_5d.transpose(0,1,2,4,3) @ Q_5d #(B,n_heads,T, W+1, Dh)
        d_repeatK = d_repeatK.reshape(B, n_kv_heads, n_rep, T, W+1, head_dim) 
        d_windows_K = nx.sum(d_repeatK, axis=2) #(B, n_kv_heads, T , W+1, head_dim)

        return 
    
    # @staticmethod
    # def _backward(gradient:nx.ArrayLike, caches:tuple[Any,...], attn_params: tuple[Any,...]) -> tuple[nx.ArrayLike,...]:
    #     fp16_x, Q,repeats_K, repeats_V, weights, output_concat = caches
    #     n_heads, head_dim, embed_dim, n_kv_heads, n_rep, Wo, freqs, Wqkv = attn_params

    #     scale = nx.float_32(nx.sqrt(head_dim))
    #     B, T, _ = fp16_x.shape
    #     d_output_concat = gradient @ Wo.T
    #     d_output = d_output_concat.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)

    #     dweights = d_output @ repeats_V.transpose(0, 1, 3, 2)
    #     d_repeatsV = weights.transpose(0, 1, 3, 2) @ d_output
    #     d_repeatsV = d_repeatsV.reshape(B, n_kv_heads, n_rep, T, head_dim)
    #     dV = d_repeatsV.sum(axis=2)

    #     dscores = softmax_derivative(weights, dweights)
    #     dscores /= scale
        
    #     dQ = dscores @ repeats_K
    #     d_repeatsK = dscores.transpose(0, 1, 3, 2) @ Q
    #     d_repeatsK = d_repeatsK.reshape(B, n_kv_heads, n_rep, T, head_dim)
    #     dK = d_repeatsK.sum(axis=2)

    #     dQ = rope_inverse(dQ, freqs)
    #     dK = rope_inverse(dK, freqs)

    #     dQ = dQ.transpose(0, 2, 1, 3).reshape(B, T, embed_dim)
    #     dK = dK.transpose(0, 2, 1, 3).reshape(B, T, n_kv_heads * head_dim)
    #     dV = dV.transpose(0, 2, 1, 3).reshape(B, T,  n_kv_heads * head_dim)

    #     dQKV = nx.concatenate([dQ, dK,dV], axis=-1)
    #     DQKV = dQKV.reshape(-1, embed_dim + 2 * (n_kv_heads * head_dim))

    #     X = fp16_x.astype(nx.float32).reshape(-1, embed_dim)
    #     dWqkv = DQKV.T @ X

    #     H = output_concat.reshape(-1, embed_dim)
    #     G = gradient.reshape(-1, embed_dim)

    #     dWo = H.T @ G
    #     dx = dQKV @ Wqkv

    #     return dx,dWqkv,dWo
        
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
        