from engine.losses import cross_entropy_gradient, cross_entropy
from engine.activations import softmax
from engine.embedding import Embedding
from engine.dataloader import DataLoader
from engine.optimizer import  AdamW
from engine.transformer_block import TransformerBlock
import engine.backend as nx
from typing import Any
import time
# FLUSH_EVERY = 32

class Transformer:
    def __init__(self, vocab_size:int, embed_dim:int,optimizer=None) -> None:
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.blocks = []
        if optimizer is None:
            optimizer = AdamW()
        self.optimizer = optimizer
        
    def __repr__(self) -> str:
        # str_layers = ""
        # for idx,i in enumerate(self.layers):
        #     str_layers += f"layer{idx}:{str(len(i))} neurons"  + "\n"
        # str_layers += f"total layer: {len(self.layers)}, params: {self.count_params_model()}" 
        return "no"

    def add_block(self, block:TransformerBlock):
        """add a transformer block"""
        self.blocks.append(block)

    def add_blocks(self, blocks:list[TransformerBlock]):
        """add transformer blocks"""
        self.blocks.extend(blocks)

    @classmethod
    def build(cls, input_size:int, output_size:int, hidden_layer_size:int=1, base_width:int=512) -> "Transformer":
        '''
        deprecated.
        Args:
            n
        build
        '''
        raise DeprecationWarning("no")
        
    def forward(self, inputs:Any, embedding:Embedding, return_cache= True, is_training=True) -> Any:
        '''
        inputs = list of inputs
        '''
        output = inputs.astype(nx.float16)
        all_masks = []
        all_caches = []

        for block in self.blocks:
            # ff_out,masks, caches = block.forward(output, is_training)
            B,T,_ = output.shape
            Wqkv = block.attention.Wqkv
            Wo = block.attention.Wo
            Wcombined = block.ff.Wcombined
            Wout = block.ff.Wout
            epsilon = block.rmsnorm1.epsilon
            gamma1 = block.rmsnorm1.gamma
            gamma2 = block.rmsnorm2.gamma
            if block.causal_mask is None or block.causal_mask.shape[0] != T:
                block.causal_mask = nx.triu(nx.ones((T, T), dtype=nx.bool_), k=1)
            ff_out ,masks, caches = block._forward(output, block.causal_mask, self.embed_dim, block.n_heads,block.n_kv_heads, block.n_rep ,block.head_dim, 
                                                   block.freqs, Wqkv, Wo, Wcombined, block.hidden_width, Wout, epsilon, gamma1, gamma2, 0.1, is_training)

            output = ff_out
            all_masks.append(masks)
            all_caches.append(caches)

        last_output =output.astype(nx.float32)
        scores = last_output @ embedding.lookup_table.T

        if return_cache:
            return scores, last_output, all_masks, all_caches
        return scores
    
    def backward(self, err_signal:Any, lookup_table, last_output, all_masks,all_caches) -> Any:
        '''
        Args:
            traces error contribution and then optimize
        '''
        current_grad = err_signal
        for block, masks,caches in zip(self.blocks[::-1], all_masks[::-1],all_caches[::-1]):
            caches_attn, caches_ff, caches_rmsnorm1, caches_rmsnorm2 = caches
            mask1, mask2 = masks
            d_attn_params = (block.n_heads, block.head_dim, block.embed_dim, block.n_kv_heads, block.n_rep,block.attention.Wo, block.freqs, block.attention.Wqkv)
            ff_params = (block.ff.Wout, block.ff.Wcombined)
            dx, dWout, dWcombined,dWqkv,dWo, d_gamma1, d_gamma2 = block._backward(current_grad, mask1=mask1, mask2=mask2, 
                                                                caches_attn=caches_attn, caches_ff=caches_ff, caches_rmsnorm1=caches_rmsnorm1, caches_rmsnorm2=caches_rmsnorm2, 
                                                                d_attn_params=d_attn_params, gamma1=block.rmsnorm1.gamma, gamma2=block.rmsnorm2.gamma, ff_params=ff_params)
            
            block.ff.dWout = dWout
            block.ff.dWcombined = dWcombined
            
            block.attention.dWqkv=dWqkv
            block.attention.dWo = dWo

            block.rmsnorm1.d_gamma = d_gamma1
            block.rmsnorm2.d_gamma = d_gamma2
            current_grad = dx
        
        return current_grad
    
    def train(self, dataloader:DataLoader, embedding:Embedding, batch_size:int=32):
        '''
        Args:
            dataloader: Dataloader object
            embedding: Embedding object
            batch_size = number of batch
        '''
        total_loss = nx.float_32(0.0)
        count = 0
        for contexts, next_tokens in dataloader.get_pairs(batch_size):              
            embedded = embedding.forward(contexts)  # shape (batch, context_size, embed_dim)
            batch_scores, last_output, all_masks, all_caches = self.forward(embedded, embedding)
            loss = cross_entropy(batch_scores, next_tokens)
            loss = nx.mean(loss)
            batch_gradient = cross_entropy_gradient(batch_scores, next_tokens)
            batch_gradient /= (batch_gradient.shape[0] * batch_gradient.shape[1])

            block_gradient = batch_gradient @ embedding.lookup_table
            d_table = batch_gradient.reshape(-1, self.vocab_size).T @ last_output.reshape(-1, self.embed_dim) 
            
            current_grad = self.backward(block_gradient, embedding.lookup_table, last_output, all_masks, all_caches)

            embedding_gradient = nx.zeros_like(embedding.lookup_table, dtype=nx.float32)
            embedding_gradient = nx.add_at(embedding_gradient, contexts, current_grad)

            total_embedding_gradient = embedding_gradient + d_table
            optimized = self.optimizer.step_many([("embedding",embedding.lookup_table, total_embedding_gradient)])  
            embedding.lookup_table = optimized["embedding"]

            all_network_params = []
            for i,block in enumerate(self.blocks):
                all_network_params.extend(
                    [(f"Wqkv_{i}", block.attention.Wqkv, block.attention.dWqkv),
                    (f"Wo_{i}", block.attention.Wo, block.attention.dWo),
                    (f"ff_wcombined_{i}", block.ff.Wcombined, block.ff.dWcombined),
                    (f"ff_wout_{i}", block.ff.Wout, block.ff.dWout),
                    (f"rmsnorm1_gamma_{i}", block.rmsnorm1.gamma, block.rmsnorm1.d_gamma),
                    (f"rmsnorm2_gamma_{i}", block.rmsnorm2.gamma, block.rmsnorm2.d_gamma)])
            
            optimized = self.optimizer.step_many(all_network_params)
            for i,block in enumerate(self.blocks):
                block.attention.Wqkv = optimized[f"Wqkv_{i}"]
                block.attention.Wo = optimized[f"Wo_{i}"]
                block.ff.Wcombined = optimized[f"ff_wcombined_{i}"]
                block.ff.Wout = optimized[f"ff_wout_{i}"]
                block.rmsnorm1.gamma = optimized[f"rmsnorm1_gamma_{i}"]
                block.rmsnorm2.gamma = optimized[f"rmsnorm2_gamma_{i}"]

            total_loss += loss.item() * next_tokens.size
            count += next_tokens.size

            del embedded, batch_scores, all_caches,all_masks, last_output, current_grad, d_table, all_network_params, optimized
            
        final_loss = total_loss / count
        return nx.float_32(final_loss)
    
    def benchmark(self, dataloader:DataLoader, embedding:Embedding, batch_size:int=32, pass_ =1):
        total_loss = nx.float_32(0.0)
        count = 0
        batch_idx = 0
        for contexts, next_tokens in dataloader.get_pairs(batch_size):  
            if batch_idx == pass_:
                break            
            embedded = embedding.forward(contexts)  # shape (batch, context_size, embed_dim)
            batch_scores, last_output, all_masks, all_caches = self.forward(embedded, embedding)
            loss = cross_entropy(batch_scores, next_tokens)
            loss = nx.mean(loss)
            # start = time.perf_counter()
            nx.eval(loss)
            # end = time.perf_counter()
            # print(f"eval loss {end-start:.3f}")
            batch_gradient = cross_entropy_gradient(batch_scores, next_tokens)
            batch_gradient /= (batch_gradient.shape[0] * batch_gradient.shape[1])

            block_gradient = batch_gradient @ embedding.lookup_table
            d_table = batch_gradient.reshape(-1, self.vocab_size).T @ last_output.reshape(-1, self.embed_dim) 
            
            current_grad = self.backward(block_gradient, embedding.lookup_table, last_output, all_masks, all_caches)
            # start = time.perf_counter()
            nx.eval(current_grad)
            # end = time.perf_counter()
            # print(f"eval current_grad {end-start:.3f}")
            embedding_gradient = nx.zeros_like(embedding.lookup_table, dtype=nx.float32)
            embedding_gradient = nx.add_at(embedding_gradient, contexts, current_grad)

            total_embedding_gradient = embedding_gradient + d_table
            optimized = self.optimizer.step_many([("embedding",embedding.lookup_table, total_embedding_gradient)])  
            embedding.lookup_table = optimized["embedding"]
            # start = time.perf_counter()
            nx.eval(embedding.lookup_table)
            # end = time.perf_counter()
            # print(f"eval embedding {end-start:.3f}")

            all_network_params = []
            for i,block in enumerate(self.blocks):
                all_network_params.extend(
                    [(f"Wqkv_{i}", block.attention.Wqkv, block.attention.dWqkv),
                    (f"Wo_{i}", block.attention.Wo, block.attention.dWo),
                    (f"ff_wcombined_{i}", block.ff.Wcombined, block.ff.dWcombined),
                    (f"ff_wout_{i}", block.ff.Wout, block.ff.dWout),
                    (f"rmsnorm1_gamma_{i}", block.rmsnorm1.gamma, block.rmsnorm1.d_gamma),
                    (f"rmsnorm2_gamma_{i}", block.rmsnorm2.gamma, block.rmsnorm2.d_gamma)])
            
            optimized = self.optimizer.step_many(all_network_params)
            for i,block in enumerate(self.blocks):
                block.attention.Wqkv = optimized[f"Wqkv_{i}"]
                block.attention.Wo = optimized[f"Wo_{i}"]
                block.ff.Wcombined = optimized[f"ff_wcombined_{i}"]
                block.ff.Wout = optimized[f"ff_wout_{i}"]
                block.rmsnorm1.gamma = optimized[f"rmsnorm1_gamma_{i}"]
                block.rmsnorm2.gamma = optimized[f"rmsnorm2_gamma_{i}"]
            
            to_eval = []
            for block in self.blocks:
                to_eval.append(block.attention.Wqkv)
                to_eval.append( block.attention.Wo)
                to_eval.append(block.ff.Wcombined)
                to_eval.append(block.ff.Wout)
                to_eval.append(block.rmsnorm1.gamma )
                to_eval.append(block.rmsnorm2.gamma )

            # start = time.perf_counter()
            nx.eval(*to_eval)
            # end = time.perf_counter()
            # print(f"eval network {end-start:.3f}")

            total_loss += loss.item() * next_tokens.size
            count += next_tokens.size

            del embedded, batch_scores, all_caches,all_masks, last_output, current_grad, d_table, all_network_params, optimized
            
            batch_idx += 1
        final_loss = total_loss / count
        return nx.float_32(final_loss)
    
    def validate(self, embedding,dataloader:DataLoader, batch_size, train_split=.9):
        total_loss = nx.float_32(0.0)
        count = 0
        dataloader.train_split = train_split
    
        for contexts, next_tokens in dataloader.get_validation_pairs(batch_size):
            embedded = embedding.forward(contexts) 
            batch_validation_scores = self.forward(embedded, embedding, False, False)
            
            val_loss = cross_entropy(batch_validation_scores, next_tokens)
            val_loss = nx.mean(val_loss)
            total_loss += val_loss.item() * next_tokens.size
            count += next_tokens.size
            
        final_loss = total_loss / count
        return nx.float_32(final_loss)

    def to_dict(self) -> dict:
        """
        get dictionary
        """
        transformer = {
            "vocab_size":self.vocab_size,
            "embed_dim": self.embed_dim,
            "blocks": [],
        }
        for block in self.blocks:
            transformer["blocks"].append(block.to_dict())
        # transformer["classifier"] = self.classifier.to_dict()
        return transformer
    
    @classmethod
    def from_dict(cls,thing:dict) -> "Transformer":
        """init from dictionary"""
        vocab_size = thing["vocab_size"]
        embed_dim = thing["embed_dim"]
        blocks = thing["blocks"]
        transformer = cls(vocab_size, embed_dim)
        for block in blocks:
            transformer.add_block(TransformerBlock.from_dict(block))
        return transformer
       
    def inference(self, context:Any, max_cache_len, embedding:Embedding, all_caches = None,  position = 0) -> Any:
        if all_caches is None:
            all_caches = [(None, None) for _ in range(len(self.blocks))]
        output = embedding.forward(context)
        for idx, block in enumerate(self.blocks):
            cached_k, cached_v = all_caches[idx]
            ff_out, cache_k, cache_v = block.inference_forward(output,max_cache_len, cached_k, cached_v, position)
            all_caches[idx] = (cache_k, cache_v)
            output = ff_out

        scores = output @ embedding.lookup_table.T
        return scores, all_caches