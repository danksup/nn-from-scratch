import engine.backend as nx
from engine.activations import softmax
from engine.activations import swish,swish_derivative

class MoE:
    def __init__(self, n_experts, embed_dim, hidden_width) -> None:
        self.hidden_width = hidden_width
        self.embed_dim = embed_dim
        self.n_experts = n_experts #E

        init = nx.sqrt(6/(embed_dim + n_experts)) 
        self.router = nx.uniform(-init,init,(self.embed_dim, self.n_experts), dtype=nx.float16)
        self.d_router = None

        scale = nx.sqrt(6 / (embed_dim+hidden_width))
        self.Wcombined = nx.uniform(-scale,scale, (n_experts, hidden_width * 2, embed_dim),dtype=nx.float16)
        self.Wout = nx.uniform(-scale,scale, (n_experts, hidden_width, embed_dim), dtype=nx.float16)

        self.dWcombined = None
        self.dWout = None

    @staticmethod
    def forward(x:nx.ArrayLike, router:nx.ArrayLike, n_expert:int,H:int, Wcombined, Wout):
        #routing
        x = x #type:ignore
        B, T, D = x.shape 
        flatten_x = x.reshape(-1, D)
        scores =  flatten_x @ router #(B*T, E)
        router_prob = softmax(scores, -1) #(B*T, E)
        #tip-1
        chosen_expert_idx = nx.argmax(router_prob, -1) #(B*T,)

        #dispatch
        sorted_indices = nx.argsort(chosen_expert_idx)
        sorted_expert_idx = chosen_expert_idx[sorted_indices]
        sorted_x = flatten_x[sorted_indices]
        counts = nx.zeros(n_expert, nx.int32)
        counts = nx.add_at(counts, sorted_expert_idx, 1)
        # counts = nx.pad(counts, (1,0), 'constant', 0)

        row_idx = nx.arange(router_prob.shape[0])
        chosen_gate = router_prob[row_idx,chosen_expert_idx] #(B*T,)
        sorted_gate = chosen_gate[sorted_indices] #(B*T,)

        cum = nx.cumsum(counts)
        cum = nx.pad(cum, (1, 0), "constant", 0)
        sorted_output = nx.zeros((B*T, D))
        expert_caches = []
        for e in range(n_expert):
            start = cum[e]
            end = cum[e+1]
            if start==end:
                expert_caches.append(None)
                continue
            token = sorted_x[start:end] #(n of token for this expert, D)
            projected = token @ Wcombined[e].T #(n of token for this expert, 2*H)
            gate_half = projected[..., :H]
            value_half = projected[..., H:]
            hidden = swish(gate_half) * value_half #(n of token for this expert, H)
            output = hidden @ Wout[e] #(n of token for this expert, D)
            sorted_output[start:end] = output * sorted_gate[start:end][...,None]
            expert_caches.append((token, projected, hidden, output))
        
        final_output = nx.zeros_like(flatten_x)
        final_output[sorted_indices] = sorted_output
        final_output = final_output.reshape(B,T,D)
        caches = (x, flatten_x, router_prob, chosen_expert_idx, sorted_expert_idx, sorted_indices, cum, sorted_gate, expert_caches)
        return final_output, caches

    @staticmethod
    def backward():
        pass
        
    