from engine.backend import nx
from engine.activations import softmax, softmax_derivative
from typing import Any

class AttentionLayer:
    def __init__(self,embed_dim:int, n_heads:int) -> None:
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        assert embed_dim % n_heads == 0
        self.head_dim = embed_dim // n_heads
        self.scale = nx.float_32(nx.sqrt(self.head_dim))
        
        scale = nx.float_32(1/self.scale)
        self.Wq = nx.uniform(-scale, scale, (embed_dim,embed_dim), dtype=nx.float32) #query, current context
        self.Wk = nx.uniform(-scale, scale, (embed_dim,embed_dim),dtype=nx.float32) #key, what information is contained
        self.Wv = nx.uniform(-scale, scale, (embed_dim,embed_dim),dtype=nx.float32) #value, what information should be sent
        self.Wo = nx.uniform(-scale,scale, (embed_dim,embed_dim), dtype=nx.float32) #projection
       
    def forward(self, x:Any) -> Any: #x shape (batch_size, context_size, embed_dim)
        self.x = x
        self.Q = x @ self.Wq 
        self.K = x @ self.Wk 
        self.V = x @ self.Wv 

        B, T, _ = x.shape
        self.Q = self.Q.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        self.K = self.K.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        self.V = self.V.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        self.scores = self.Q @ self.K.transpose(0, 1, 3, 2)
        self.scores /= self.scale

        mask = nx.triu(nx.ones((T, T), dtype=nx.bool), k=1)
        self.scores = nx.where(mask, nx.float_32(-1e9), self.scores)
        self.weights = softmax(self.scores)
        self.output = self.weights @ self.V
        self.output_concat = self.output.transpose(0, 2, 1, 3).reshape(B, T, self.embed_dim)
        self.output_projected = self.output_concat @ self.Wo
        return self.output_projected
    
    def backward(self, output_gradient) -> Any:
        d_output_concat = output_gradient @ self.Wo.T
        B, T, _ = self.x.shape
        d_output = d_output_concat.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        dweights = d_output @ self.V.transpose(0, 1, 3, 2)
        self.dV = self.weights.transpose(0, 1, 3, 2) @ d_output
        dscores = softmax_derivative(self.weights, dweights)
        dscores /= self.scale
        self.dQ = dscores @ self.K
        self.dK = dscores.transpose(0, 1, 3, 2) @ self.Q

        dQ = self.dQ.transpose(0, 2, 1, 3).reshape(B, T, self.embed_dim)
        dK = self.dK.transpose(0, 2, 1, 3).reshape(B, T, self.embed_dim)
        dV = self.dV.transpose(0, 2, 1, 3).reshape(B, T, self.embed_dim)

        self.dWq = nx.sum(self.x.transpose(0,2,1) @ dQ, axis=0, dtype=nx.float32)
        self.dWk = nx.sum(self.x.transpose(0,2,1) @ dK, axis=0, dtype=nx.float32)
        self.dWv = nx.sum(self.x.transpose(0,2,1) @ dV, axis=0, dtype=nx.float32)
        self.dWo = nx.sum(self.output_concat.transpose(0,2,1)@ output_gradient,axis=0,dtype=nx.float32)

        dx = dQ @ self.Wq.T + dK @ self.Wk.T + dV @ self.Wv.T
        return dx
        
    def to_dict(self):
        return {
            "embed_dim":self.embed_dim,
            "n_heads":self.n_heads,
            "Wq":self.Wq.tolist(),
            "Wk":self.Wk.tolist(),
            "Wv":self.Wv.tolist(),
            "Wo":self.Wo.tolist(),
        }
    
    @classmethod
    def from_dict(cls,thing) -> "AttentionLayer":
        embed_dim = thing["embed_dim"]
        n_heads = thing["n_heads"]
        Wq = thing["Wq"]
        Wk = thing["Wk"]
        Wv = thing["Wv"]
        Wo = thing["Wo"]

        attention = cls(embed_dim,n_heads)
        attention.Wk = nx.array(Wk, dtype=nx.float32)
        attention.Wq = nx.array(Wq, dtype=nx.float32)
        attention.Wv = nx.array(Wv, dtype=nx.float32)
        attention.Wo = nx.array(Wo, dtype=nx.float32)

        return attention
        

        