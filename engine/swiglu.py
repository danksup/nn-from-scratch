from engine.backend import nx
from engine.activations import swish,swish_derivative

class SwiGLU:
    def __init__(self, hidden_width:int, embed_dim) -> None:
        self.hidden_width = hidden_width
        self.embed_dim = embed_dim
        scale = nx.float_32(1.0/nx.sqrt(embed_dim))
        self.Wgate = nx.uniform(-scale,scale, (hidden_width,embed_dim), dtype=nx.float16)
        self.Wvalue = nx.uniform(-scale,scale, (hidden_width,embed_dim), dtype=nx.float16)
        self.Wout = nx.uniform(-scale,scale, (hidden_width,embed_dim), dtype=nx.float16)

    def forward(self, x):
        self.x = x
        self.gate_linear = x @ self.Wgate.T
        self.gate = swish(self.gate_linear)
        self.value = x @ self.Wvalue.T
        self.hidden = self.gate * self.value

        self.output = self.hidden @ self.Wout
        return self.output
    
    def backward(self, gradient):
        batch_size, seq_len, _ = self.x.shape

        X = self.x.reshape(-1, self.x.shape[-1])                 # (B*S, embed)
        H = self.hidden.reshape(-1, self.hidden.shape[-1])       # (B*S, hidden)
        G = gradient.reshape(-1, gradient.shape[-1])             # (B*S, embed)

        self.dWout = (H.T @ G) / (batch_size * seq_len)
        d_hidden = gradient @ self.Wout.T

        d_gate = d_hidden * self.value * swish_derivative(self.gate_linear)
        d_value = d_hidden * self.gate

        DG = d_gate.reshape(-1, d_gate.shape[-1])      # (B*S, hidden)
        DV = d_value.reshape(-1, d_value.shape[-1])    # (B*S, hidden)

        self.dWgate = (DG.T @ X) / (batch_size * seq_len)
        self.dWvalue = (DV.T @ X) / (batch_size * seq_len)

        dx = d_gate @ self.Wgate + d_value @ self.Wvalue

        return dx
    
    def to_dict(self) -> dict:
        return {
            "configs":(self.hidden_width,self.embed_dim),
            "Wgate":self.Wgate.tolist(),
            "Wvalue":self.Wvalue.tolist(),
            "Wout":self.Wout.tolist(),
        }
    
    @classmethod
    def from_dict(cls, thing) -> "SwiGLU":
        hidden_width, embed_dim = thing["configs"]
        swiglu = cls(hidden_width, embed_dim)
        swiglu.Wgate = nx.array(thing["Wgate"], dtype=nx.float16)
        swiglu.Wvalue = nx.array(thing["Wvalue"], dtype=nx.float16)
        swiglu.Wout = nx.array(thing["Wout"], dtype=nx.float16)

        return swiglu
