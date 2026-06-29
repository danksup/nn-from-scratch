from engine.backend import nx
from engine.activations import swish,swish_derivative

class SwiGLU:
    def __init__(self, hidden_width:int, embed_dim) -> None:
        self.hidden_width = hidden_width
        self.embed_dim = embed_dim
        scale = nx.float_32(1.0/nx.sqrt(embed_dim))
        # self.Wgate = nx.uniform(-scale,scale, (hidden_width,embed_dim), dtype=nx.float16)
        # self.Wvalue = nx.uniform(-scale,scale, (hidden_width,embed_dim), dtype=nx.float16)
        self.Wcombined = nx.uniform(-scale,scale, (hidden_width * 2,embed_dim),dtype=nx.float16)
        self.Wout = nx.uniform(-scale,scale, (hidden_width,embed_dim), dtype=nx.float16)
        
        self.dWcombined = None
        self.dWout = None
    @staticmethod
    def _forward(x, hidden_width, Wcombined, Wout):
        fp16_x = x.astype(nx.float16)
        combined_linear = fp16_x @ Wcombined.T
        gate_linear = combined_linear[...,:hidden_width]
        value = combined_linear[..., hidden_width:]
        gate = swish(gate_linear)
        hidden = gate * value
        output = hidden @ Wout
        cache = (fp16_x, gate_linear, gate, value, hidden)
        return output,cache
    
    @staticmethod
    def _backward(gradient, caches, ff_params):
        x, gate_linear, gate, value, hidden = caches
        Wout, Wcombined = ff_params
        batch_size, seq_len, _ = x.shape

        X = x.reshape(-1, x.shape[-1])
        H = hidden.reshape(-1, hidden.shape[-1])
        G = gradient.reshape(-1, gradient.shape[-1]) 
        dWout = (H.T @ G) / (batch_size * seq_len)
        d_hidden = gradient @ Wout.T

        d_gate =  d_hidden * value * swish_derivative(gate_linear)
        d_value = d_hidden * gate
        d_combined = nx.concatenate([d_gate, d_value], axis=-1)
        DC = d_combined.reshape(-1, d_combined.shape[-1])
        dWcombined = DC.T @ X / (batch_size * seq_len)

        dx = d_combined @ Wcombined

        return dx, dWout, dWcombined
    
    def to_dict(self) -> dict:
        return {
            "configs":(self.hidden_width,self.embed_dim),
            "Wcombine":self.Wcombined.tolist(),
            "Wout":self.Wout.tolist(),
        }
    
    @classmethod
    def from_dict(cls, thing) -> "SwiGLU":
        hidden_width, embed_dim = thing["configs"]
        swiglu = cls(hidden_width, embed_dim)
        swiglu.Wcombined = nx.array(thing["Wcombine"], dtype=nx.float16)
        swiglu.Wout = nx.array(thing["Wout"], dtype=nx.float16)

        return swiglu
