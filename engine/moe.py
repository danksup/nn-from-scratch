import engine.backend as nx
from engine.activations import softmax

class MoE:
    def __init__(self, n_experts, embed_dim, hidden_width) -> None:
        self.hidden_width = hidden_width
        self.embed_dim = embed_dim
        self.n_experts = n_experts #E

        init = nx.sqrt(6/(embed_dim + n_experts)) #xavier init
        self.router = nx.uniform(-init,init,(self.embed_dim, self.n_experts))
        self.d_router = None

        scale = nx.sqrt(6 / (embed_dim+hidden_width))
        self.Wcombined = nx.uniform(-scale,scale, (n_experts, hidden_width * 2, embed_dim),dtype=nx.float16)
        self.Wout = nx.uniform(-scale,scale, (n_experts, hidden_width, embed_dim), dtype=nx.float16)

        self.dWcombined = None
        self.dWout = None

    @staticmethod
    def forward(x:nx.ArrayLike, router:nx.ArrayLike):
        #routing
        x = x.astype(nx.float16) #type:ignore
        B, T, D = x.shape 
        flatten_x = x.reshape(-1, D)
        scores =  flatten_x @ router #shape = (B*T, E)
        router_prob = softmax(scores, -1)
        expert_idx = nx.argmax(router_prob, -1)

        #dispatch
        

    @staticmethod
    def backward():
        pass
        
    