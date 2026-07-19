import engine.backend as nx
from engine.activations import softmax, softmax_derivative
from engine.rope import rope_forward, rope_inverse
from typing import Any
import time

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
    def _forward(x:nx.ArrayLike, causal_mask:nx.ArrayLike, embed_dim:int, n_kv_heads:int, n_heads:int, n_rep:int, head_dim:int, W, freqs:int, Wqkv:nx.ArrayLike, Wo:nx.ArrayLike):
        combined = x @ Wqkv.T 
        SCALE = nx.float_32(nx.sqrt(head_dim))

        Q = combined[..., :embed_dim] #shape: (B, T, D)
        K = combined[..., embed_dim: embed_dim + (n_kv_heads * head_dim)]  #shape: (B, T, n_kv_heads * H)
        V = combined[..., embed_dim + (n_kv_heads * head_dim):] #shape: (B, T, n_kv_heads * H)
        
        B, T, _ = x.shape
        Q = Q.reshape(B, T, n_heads, head_dim).transpose(0,2,1,3) #(B,n_heads,T, Dh)
        K = K.reshape(B, T, n_kv_heads, head_dim).transpose(0,2,1,3) #(B, n_kv_heads, T, Dh)
        V = V.reshape(B, T, n_kv_heads, head_dim).transpose(0,2,1,3) #(B, n_kv_heads, T, Dh)

        Q = rope_forward(Q, freqs)
        K = rope_forward(K, freqs)

        pad = [(0,0), (0,0),(W,0), (0,0)]
        P = T + W
        stride = (head_dim * n_kv_heads * P, P * head_dim, head_dim, head_dim, 1)
        padded_K = nx.pad(K, pad, constant_value=0)  #(B, n_kv_head, T+W, Dh)
        padded_V = nx.pad(V, pad, constant_value=0) #(B,n_kv_head,T+W, Dh)
        shape = (padded_K.shape[0], padded_K.shape[1], T, W + 1, head_dim) #(B, n_kv_head, T, W+1, Dh)

        windows_K = nx.as_strided(padded_K, shape=shape,strides=stride) #shape=shape
        windows_V = nx.as_strided(padded_V, shape=shape,strides=stride) #shape=shape

        Q = Q.reshape(B, n_kv_heads, n_rep, T, head_dim)

        # start = time.perf_counter()
        Q_6d = Q[:,:,:,:,None,:] #(B, n_kv_heads, n_rep, T,1, Dh)
        windows_K_6d = windows_K[:,:,None,:,:,:] #(B, n_kv_head, 1, T, W+1, Dh)
        scores = Q_6d @ windows_K_6d.transpose(0,1,2,3,5,4) #B, n_kv_heads, n_rep, T, 1, W+1
        # scores = nx.einsum("bkrtd,bktwd->bkrtw",Q, windows_K) #(B, n_kv_heads, n_rep, T, W+1)
        # nx.eval(scores)  
        # end = time.perf_counter()
        # print(f"scores {end-start:.5f}")

        # print("swa scores unscaled",scores.dtype)
        scores = scores[:,:,:,:,0,:].reshape(B, -1, T, W+1)
        scores /= SCALE
        scores = nx.where(causal_mask, -1e9, scores)
        weights = softmax(scores) #(B, n_heads, T, W+1)
        weights = weights.reshape(B, n_kv_heads, n_rep, T, W+1)

        # start = time.perf_counter()
        weights_6d = weights[:,:,:,:,None,:]     #(B, n_kv_head, n_rep, T, 1, W+1)
        windows_V_6d = windows_V[:,:,None,:,:,:] #(B, n_kv_head, 1, T, W+1, Dh)
        # output = nx.einsum("bkrtw,bktwd->bkrtd", weights, windows_V) #(B, n_kv_heads, n_rep, T, Dh)
        output = weights_6d @ windows_V_6d #(B,K,R,T,1,D)
        # nx.eval(output)  
        # end = time.perf_counter()
        # print(f"output {end-start:.5f}")
        
        output = output[:,:,:,:,0,:]
        output_concat = output.transpose(0, 3, 1, 2, 4).reshape(B, T, embed_dim)
        output_projected = output_concat @ Wo #B,T,D
        cache = (x, Q, windows_K, windows_V, weights, output_concat)
        return output_projected, cache
    
    @staticmethod
    def _backward(gradient:nx.ArrayLike, caches:tuple[Any,...], attn_params: tuple[Any,...]) :#-> tuple[nx.ArrayLike,...]:
        fp16_x, Q, windows_K, windows_V, weights, output_concat = caches
        n_heads, head_dim, embed_dim, n_kv_heads, n_rep, W, Wo, freqs, Wqkv = attn_params

        scale = nx.float_32(nx.sqrt(head_dim))
        B, T, D = fp16_x.shape
        d_output_concat = nx.einsum("btd,fd->btf",gradient, Wo) #(B,T,D)
        d_output = d_output_concat.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3) #(B, n_heads, T,  Dh)
        d_output_split = d_output.reshape(B, n_kv_heads,n_rep,T, head_dim)

        # start = time.perf_counter()
        d_output_split_6d = d_output_split[:,:,:,:,None,:] #B, n_kv_heads, n_rep, T, 1, Dh
        windows_V_6d = windows_V[:,:,None,:,:,:] #(B, n_kv_head, 1, T, W+1, Dh)
        # d_weights = nx.einsum("bkrtd,bktwd->bkrtw", d_output_split, windows_V) #(B, n_kv_heads, n_rep, T, W+1)
        d_weights = d_output_split_6d @ windows_V_6d.transpose(0,1,2,3,5,4) #B, n_kv_heads, n_rep,T, 1, W+1
        # nx.eval(d_weights)  
        # end = time.perf_counter()
        # print(f"d_weights {end-start:.5f}")

        # start = time.perf_counter()
        # weights_6d = weights[:,:,:,:,:,None]     #(B, n_kv_head, n_rep, T, W+1, 1)
        d_windows_V = nx.einsum("bkrtw,bkrtd->bktwd", weights, d_output_split) #(B, n_kv_head, T , W+1, Dh)
        # d_windows_V = weights_6d @ d_output_split_6d #B, n_kv_heads, n_rep, T, W+1, Dh)
        # d_windows_V = nx.sum(d_windows_V, axis=2)
        # nx.eval(d_windows_V)  
        # end = time.perf_counter()
        # print(f"d_windows_V {end-start:.5f}")

        d_weights = d_weights[:,:,:,:,0,:]
        d_scores = softmax_derivative(weights, d_weights) #(B, n_kv_heads, n_rep, T, W+1)
        d_scores /= scale

        # start = time.perf_counter()
        d_scores_6d = d_scores[:,:,:,:,None,:] #(B, n_kv_heads, n_rep, T, 1,W+1)
        windows_K_6d = windows_K[:,:,None,:,:,:] #(B, n_kv_head, 1, T, W+1, Dh)
        # dQ = nx.einsum("bkrtw,bktwd->bkrtd", d_scores, windows_K)#(B, n_kv_heads, n_rep, T, Dh)
        dQ = d_scores_6d @ windows_K_6d
        # nx.eval(dQ)  
        # end = time.perf_counter()
        # print(f"dQ {end-start:.5f}")

        dQ = dQ.reshape(B, -1, T, head_dim)

        # start = time.perf_counter()
        d_windows_K = nx.einsum("bkrtw,bkrtd->bktwd", d_scores, Q) #(B,n_kv_heads,T, W+1, Dh)
        # nx.eval(d_windows_K)  
        # end = time.perf_counter()
        # print(f"d_windows_K {end-start:.5f}")

        # start = time.perf_counter()
        d_padded_K = nx.zeros((B, n_kv_heads, T+W, head_dim))
        d_padded_V = nx.zeros((B, n_kv_heads, T+W, head_dim))
        for slot in range(W + 1):
            d_padded_K[:, :, slot:slot + T, :] += d_windows_K[:, :, :, slot, :]
            d_padded_V[:, :, slot:slot + T, :] += d_windows_V[:, :, :, slot, :]
        # nx.eval(d_padded_K, d_padded_V)  
        # end = time.perf_counter()
        # print(f"padded loop {end-start:.5f}")
            
        dK = d_padded_K[:, :, W:, :]
        dV = d_padded_V[:, :, W:, :]

        dQ = rope_inverse(dQ, freqs)
        dK = rope_inverse(dK, freqs)

        dQ = dQ.transpose(0, 2, 1, 3).reshape(B, T, embed_dim)
        dK = dK.transpose(0, 2, 1, 3).reshape(B, T, n_kv_heads * head_dim)
        dV = dV.transpose(0, 2, 1, 3).reshape(B, T,  n_kv_heads * head_dim)

        dQKV = nx.concatenate([dQ, dK,dV], axis=-1) #(B,T, D + 2 * (n_kv_heads * Dh))
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
            "W":self.W,
            "Wqkv":self.Wqkv.tolist(),
            "Wo":self.Wo.tolist(),
        }
    
    @classmethod
    def from_dict(cls,thing) -> "AttentionLayer":
        """deserialize"""
        embed_dim = thing["embed_dim"]
        n_kv_heads = thing["n_kv_heads"]
        n_heads = thing["n_heads"]
        W = thing["W"]
        Wqkv = thing["Wqkv"]
        Wo = thing["Wo"]

        attention = cls(embed_dim,n_heads, n_kv_heads, W)
        attention.Wqkv = nx.array(Wqkv, dtype=nx.float16)
        attention.Wo = nx.array(Wo, dtype=nx.float16)

        return attention
        