from engine.backend import Backend
from engine.activations import softmax, softmax_derivative
from typing import Any
nx = Backend()
class AttentionLayer:
    def __init__(self, embed_dim:int) -> None:
        self.embed_dim = embed_dim
        self.scale = nx.float_32(nx.sqrt(embed_dim))
        
        scale = nx.float_32(1/self.scale)
        self.Wq = nx.uniform(-scale, scale, (embed_dim,embed_dim), dtype=nx.float32) #query, current context
        self.Wk = nx.uniform(-scale, scale, (embed_dim,embed_dim),dtype=nx.float32) #key, what information is contained
        self.Wv = nx.uniform(-scale, scale, (embed_dim,embed_dim),dtype=nx.float32) #value, what information should be sent
        self.Bq =  nx.uniform(-scale, scale, (embed_dim,),dtype=nx.float32)
        self.Bk =  nx.uniform(-scale, scale, (embed_dim,),dtype=nx.float32)
        self.Bv =  nx.uniform(-scale, scale, (embed_dim,),dtype=nx.float32)



    def forward(self, x:Any) -> Any: #x shape (batch_size, context_size, embed_dim)
        self.x = x
        self.Q = x @ self.Wq + self.Bq
        self.K = x @ self.Wk + self.Bk
        self.V = x @ self.Wv + self.Bv

        self.scores = self.Q @ self.K.transpose(0,2,1)
        self.scores /= self.scale

        mask = nx.triu(nx.ones((self.scores.shape[1],self.scores.shape[2]), dtype=nx.bool), k = 1)
        self.scores = nx.where(mask, nx.float_32(-1e9), self.scores)
        self.weights = softmax(self.scores)
        self.output = self.weights @ self.V
    
        return self.output
    
    def backward(self, output_gradient) -> Any:

        dweights = (output_gradient@ self.V.transpose(0,2,1) )

        dscores = softmax_derivative(self.weights,dweights)
        dscores /= self.scale

        self.dV = (self.weights.transpose(0,2,1)@ output_gradient)
        self.dQ = dscores @ self.K
        self.dK = (dscores.transpose(0,2,1)@ self.Q)
        self.dWq = nx.sum(self.x.transpose(0,2,1)@ self.dQ,axis=0, dtype=nx.float32)
        self.dWk = nx.sum(self.x.transpose(0,2,1)@ self.dK,axis=0, dtype=nx.float32)
        self.dWv = nx.sum(self.x.transpose(0,2,1)@ self.dV,axis=0, dtype=nx.float32)

        self.dBq = nx.sum(self.dQ, axis=(0,1), dtype=nx.float32)
        self.dBk = nx.sum(self.dK, axis=(0,1), dtype=nx.float32)
        self.dBv = nx.sum(self.dV, axis=(0,1), dtype=nx.float32)

        self.dx = (self.dQ @ self.Wq.T + self.dK @ self.Wk.T +self.dV @ self.Wv.T)
    
        return self.dx
        
    def to_dict(self):
        return {
            "embed_dim":self.embed_dim,
            "Wq":self.Wq.tolist(),
            "Wk":self.Wk.tolist(),
            "Wv":self.Wv.tolist(),
            "Bv":self.Bv.tolist(),
            "Bk":self.Bk.tolist(),
            "Bq":self.Bq.tolist(),
        }
    
    @classmethod
    def from_dict(cls,thing) -> "AttentionLayer":
        embed_dim = thing["embed_dim"]
        Wq = thing["Wq"]
        Wk = thing["Wk"]
        Wv = thing["Wv"]
        Bq = thing["Bq"]
        Bk = thing["Bk"]
        Bv = thing["Bv"]

        attention = cls(embed_dim)
        attention.Wk = nx.array(Wk, dtype=nx.float32)
        attention.Wq = nx.array(Wq, dtype=nx.float32)
        attention.Wv = nx.array(Wv, dtype=nx.float32)
        attention.Bk = nx.array(Bk, dtype=nx.float32)
        attention.Bq = nx.array(Bq, dtype=nx.float32)
        attention.Bv = nx.array(Bv, dtype=nx.float32)

        return attention
        

        