import numpy as np
from engine.activations import softmax, softmax_derivative

class AttentionLayer:
    def __init__(self, embed_dim:int) -> None:
        self.embed_dim = embed_dim
        scale = 1/np.sqrt(embed_dim)
        rng = np.random.default_rng()

        self.Wq = rng.uniform(-scale, scale, (embed_dim,embed_dim)) #query, current context
        self.Wk = rng.uniform(-scale, scale, (embed_dim,embed_dim)) #key, what information is contained
        self.Wv = rng.uniform(-scale, scale, (embed_dim,embed_dim)) #value, what information should be sent
        self.Bq =  rng.uniform(-scale, scale, (embed_dim,))
        self.Bk =  rng.uniform(-scale, scale, (embed_dim,))
        self.Bv =  rng.uniform(-scale, scale, (embed_dim,))



    def forward(self, x:np.ndarray): #x shape (batch_size, context_size, embed_dim)
        self.x = x
        self.Q = x @ self.Wq + self.Bq
        self.K = x @ self.Wk + self.Bk
        self.V = x @ self.Wv + self.Bv

        self.scores = self.Q @ self.K.transpose(0,2,1) / np.sqrt(self.embed_dim)
        self.weights = softmax(self.scores)
        self.output = self.weights @ self.V

    
        return self.output
    
    def backward(self, output_gradient):

        dweights = (output_gradient@ self.V.transpose(0,2,1) )

        dscores = softmax_derivative(self.weights,dweights)
        dscores /= np.sqrt(self.embed_dim)

        self.dV = (self.weights.transpose(0,2,1)@ output_gradient)
        self.dQ = dscores @ self.K
        self.dK = (dscores.transpose(0,2,1)@ self.Q)
        self.dWq = np.sum(self.x.transpose(0,2,1)@ self.dQ,axis=0)
        self.dWk = np.sum(self.x.transpose(0,2,1)@ self.dK,axis=0)
        self.dWv = np.sum(self.x.transpose(0,2,1)@ self.dV,axis=0)

        self.dBq = np.sum(self.dQ, axis=(0,1))
        self.dBk = np.sum(self.dK, axis=(0,1))
        self.dBv = np.sum(self.dV, axis=(0,1))

        self.dx = (self.dQ @ self.Wq.T + self.dK @ self.Wk.T +self.dV @ self.Wv.T)
    
        return self.dx
        
        