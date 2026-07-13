import engine.backend as nx
from engine.activations import softmax, softmax_derivative
from engine.activations import swish,swish_derivative
import math

class MoE:
    def __init__(self, capacity_factor, n_experts, embed_dim, hidden_width) -> None:
        self.hidden_width = hidden_width
        self.embed_dim = embed_dim
        self.n_experts = n_experts #E
        self.cf = capacity_factor

        init = nx.sqrt(6/(embed_dim + n_experts)) 
        self.router = nx.uniform(-init,init,(self.embed_dim, self.n_experts), dtype=nx.float32)
        self.d_router = None

        scale = nx.sqrt(6 / (embed_dim+hidden_width))
        self.Wcombined = nx.uniform(-scale,scale, (n_experts, hidden_width * 2, embed_dim),dtype=nx.float16)
        self.Wout = nx.uniform(-scale,scale, (n_experts, hidden_width, embed_dim), dtype=nx.float16)

        self.dWcombined = None
        self.dWout = None

    @staticmethod
    def forward(x:nx.ArrayLike, capacity_factor, router:nx.ArrayLike, n_expert:int,H:int, Wcombined, Wout):
        #routing
        B, T, D = x.shape 
        N = B * T
        capacity = math.ceil(capacity_factor * N / n_expert)
        flatten_x = x.reshape(-1, D)
        scores =  flatten_x @ router #(N, E)
        router_prob = softmax(scores, -1) #(N, E)
        #tip-1
        chosen_expert_idx = nx.argmax(router_prob, -1) #(N,)

        histogram = nx.zeros(n_expert, dtype=nx.int32)
        histogram = nx.add_at(histogram,chosen_expert_idx,1) 
        avg_prob = nx.mean(router_prob, axis=0) #P
        normalized_histogram = histogram / N #f
        router_loss = n_expert * nx.sum(normalized_histogram * avg_prob) #L

        #dispatch
        routing_mask = nx.zeros((N,n_expert), nx.int32)
        row = nx.arange(N, dtype=nx.int32)
        routing_mask = nx.add_at(routing_mask, (row, chosen_expert_idx), 1)

        cum_assignment = nx.cumsum(routing_mask, axis=0, dtype=nx.int32)

        row_idx = nx.arange(router_prob.shape[0], dtype=nx.int32)
        chosen_gate = router_prob[row_idx,chosen_expert_idx] #(N,)
        slot_idx = cum_assignment[row_idx, chosen_expert_idx] - 1 #type:ignore #(N,)

        valid = slot_idx < capacity
        num_valid = nx.array(valid)
        masked_tokens = flatten_x * num_valid[:, None]
        safe_slot = nx.clip(slot_idx, 0, capacity - 1, dtype=nx.int32)
        expert_input = nx.zeros((n_expert, capacity, D))
        expert_input = nx.add_at(expert_input, (chosen_expert_idx, safe_slot), masked_tokens)

        expert_gate = nx.zeros((n_expert, capacity))
        safe_gates = nx.where(valid, chosen_gate, 0.0)
        expert_gate = nx.add_at(expert_gate, (chosen_expert_idx, safe_slot), safe_gates)

        projected = expert_input @ Wcombined.transpose(0,2,1) #(E, capacity, 2H)
        gate_half = projected[..., :H]
        value_half = projected[..., H:]
        hidden = swish(gate_half) * value_half #(E, capacity, H)
        raw_output = hidden @ Wout #(E, capacity, D)
        gated_output = raw_output * expert_gate[..., None]
        final_output = gated_output[chosen_expert_idx, safe_slot]
        final_output = final_output * num_valid[..., None]
        final_output = final_output.reshape(B,T,D)
        cache = (flatten_x, router_prob, chosen_expert_idx, valid, safe_slot, expert_input, expert_gate, projected, hidden, raw_output, normalized_histogram)
        return final_output, cache, router_loss

    @staticmethod
    def backward(gradient , caches, moe_configs, ff_params):
        flatten_x, router_prob, chosen_expert_idx, valid, safe_slot, expert_input, expert_gate, projected, hidden, raw_output, normalized_histogram = caches
        Wout, Wcombined = ff_params
        capacity_factor, n_experts, hidden_width, router = moe_configs
        B,T,D = gradient.shape
        N = B*T
        flatten_gradient = gradient.reshape(-1, D)
        capacity = math.ceil(capacity_factor * N / n_experts)
        num_valid = nx.array(valid, dtype=nx.int32)
        d_masked_output = flatten_gradient * num_valid[...,None]

        d_gated_output = nx.zeros((n_experts, capacity, D))
        d_gated_output = nx.add_at(d_gated_output, (chosen_expert_idx, safe_slot), d_masked_output)
        d_raw_output = d_gated_output * expert_gate[..., None] #(E, capacity, D)
        d_expert_gate = nx.sum(d_gated_output * raw_output, axis=-1) #(E, capacity)
       
        dWout = hidden.transpose(0, 2, 1) @ d_raw_output
        d_hidden = d_raw_output @ Wout.transpose(0, 2, 1) #(E,C,H)

        gate_half = projected[..., :hidden_width]
        value_half = projected[..., hidden_width:]

        d_gate_half = d_hidden * value_half * swish_derivative(gate_half)
        d_value_half = d_hidden * swish(gate_half)
        d_projected = nx.concatenate([d_gate_half, d_value_half], axis=-1) #(E, C, 2H)
        dWcombined = d_projected.transpose(0, 2, 1) @ expert_input #(E, 2H, D)
        d_expert_input = d_projected @ Wcombined #(E, C, D)

        row_idx = nx.arange(safe_slot.shape[0], dtype=nx.int32)
        d_x_expert = nx.zeros((N,D))
        d_x_expert[row_idx] = d_expert_input[chosen_expert_idx[row_idx], safe_slot[row_idx]]
        d_x_expert *= num_valid[...,None]

        d_chosen_gate = nx.zeros((N,))
        d_chosen_gate[row_idx] = d_expert_gate[chosen_expert_idx[row_idx], safe_slot[row_idx]]
        d_chosen_gate *= num_valid

        d_router_prob = nx.zeros((N,n_experts))
        d_router_prob[row_idx, chosen_expert_idx] = d_chosen_gate

        d_avg_prob = n_experts * normalized_histogram
        d_router_prob += 0.01 * (d_avg_prob / N)

        d_scores = softmax_derivative(router_prob, d_router_prob) #(N,E)

        d_router = flatten_x.T @ d_scores
        d_x_router = d_scores @ router.T #(N, D)
        dx_flat = d_x_expert + d_x_router

        dx = dx_flat.reshape(B,T,D)
        return dx, dWcombined, dWout, d_router

    def to_dict(self) -> dict:
        return {
            "configs":(self.cf, self.n_experts, self.hidden_width,self.embed_dim),
            "router": self.router.tolist(),
            "Wcombine":self.Wcombined.tolist(),
            "Wout":self.Wout.tolist(),
        }
    
    @classmethod
    def from_dict(cls, thing:dict) -> "MoE":
        capacity_factor, n_experts, hidden_width, embed_dim = thing["configs"]
        moe = cls(capacity_factor, n_experts, embed_dim, hidden_width)
        moe.Wcombined = nx.array(thing["Wcombine"], dtype=nx.float16)
        moe.Wout = nx.array(thing["Wout"], dtype=nx.float16)
        moe.router = nx.array(thing["router"], dtype=nx.float32)
        return moe
