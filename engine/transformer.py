from engine.swiglu import SwiGLU
from engine.losses import cross_entropy_gradient, cross_entropy
from engine.activations import softmax
from engine.embedding import Embedding
from engine.dataloader import DataLoader
from engine.optimizer import  AdamW
from engine.transformer_block import TransformerBlock
from engine.backend import nx
from typing import Any
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
        
    def forward(self, inputs:Any, embedding:Embedding, return_cache= True) -> Any:
        '''
        inputs = list of inputs
        '''
        output = inputs.astype(nx.float16)
        attention_caches = []
        ff_caches = []
        rmns1_caches = []
        rmns2_caches = []
        for block in self.blocks:
            output, attention_cache, ff_cache, rmns1_cache, rmns2_cache = block.forward(output)
            attention_caches.append(attention_cache)
            ff_caches.append(ff_cache)
            rmns1_caches.append(rmns1_cache)
            rmns2_caches.append(rmns2_cache)
        last_output =output.astype(nx.float32)

        scores = last_output @ embedding.lookup_table.T
        if return_cache:
            return scores, attention_caches, ff_caches, last_output, rmns1_caches, rmns2_caches
        return scores
    
    def backward(self, err_signal:Any, lookup_table, attention_caches:list,ff_caches:list, last_output:Any,  rmns1_caches:Any, rmns2_caches:Any) -> Any:
        '''
        Args:
            traces error contribution and then optimize
        '''
        err_signal =  err_signal.astype(nx.float32)
        block_gradient = err_signal @ lookup_table
        d_table = err_signal.reshape(-1, self.vocab_size).T @ last_output.reshape(-1, self.embed_dim) #type: ignore
        d_table /= (err_signal.shape[0]* err_signal.shape[1])
        
        current_grad = block_gradient
        for block, block_cache,ff_cache, rmns1_cache, rmns2_cache in zip(self.blocks[::-1], attention_caches[::-1], ff_caches[::-1],  rmns1_caches[::-1], rmns2_caches[::-1]):
            current_grad = block.backward(current_grad, block_cache, ff_cache, rmns1_cache, rmns2_cache)
        
        return current_grad,d_table

    def train_mode(self):
        for block in self.blocks:
            block.train()

    def eval_mode(self):
        for block in self.blocks:
            block.eval()
    
    @staticmethod
    @nx.nx.compile
    def compiled_loss(batch_scores, next_tokens):
        softmax_batch_scores = softmax(batch_scores)
        loss = nx.sum(cross_entropy(softmax_batch_scores, next_tokens), dtype=nx.float32)
        return loss, softmax_batch_scores
    
    def train(self, dataloader:DataLoader, embedding:Embedding, batch_size:int=32):
        '''
        Args:
            dataloader: Dataloader object
            embedding: Embedding object
            batch_size = number of batch
        '''
        total_loss = nx.float_32(0.0)
        count = 0
        # i = 0
        for contexts, next_tokens in dataloader.get_pairs(batch_size):  

            embedded = embedding.forward(contexts)  # shape (batch, context_size, embed_dim)
            
            batch_scores, attention_caches,ff_caches, last_output, rmns1_caches, rmns2_caches = self.forward(embedded, embedding)
            loss, softmax_batch_scores = self.compiled_loss(batch_scores, next_tokens)

            batch_gradient = cross_entropy_gradient(softmax_batch_scores, next_tokens)
            batch_gradient /= (batch_gradient.shape[0] * batch_gradient.shape[1])

            
            current_grad,d_table = self.backward(batch_gradient, embedding.lookup_table, attention_caches, ff_caches, last_output,rmns1_caches, rmns2_caches)
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

            total_loss += loss.item() 
            count += next_tokens.size

            del embedded, batch_scores, attention_caches, last_output, current_grad, d_table, ff_caches, all_network_params, optimized
            
            # if i % FLUSH_EVERY == 0:
            #     nx.eval(loss, embedding.lookup_table, *optimized.values(),
            #             *[w for block in self.blocks
            #                 for w in (block.attention.Wq, block.attention.Wk, block.attention.Wv, block.attention.Wo,block.ff.Wgate, block.ff.Wvalue,
            #                       block.ff.Wout, block.rmsnorm1.gamma, block.rmsnorm2.gamma)])
                
            #     self.optimizer.eval_state()
            # i = i + 1
        final_loss = total_loss / count
        return nx.float_32(final_loss)
    
    def validate(self, embedding,dataloader:DataLoader, batch_size, train_split=.9):
        total_loss = nx.float_32(0.0)
        count = 0
        self.eval_mode()
        dataloader.train_split = train_split
    
        for contexts, next_tokens in dataloader.get_validation_pairs(batch_size):
            embedded = embedding.forward(contexts) 
            batch_validation_scores = self.forward(embedded, embedding, False)
            softmax_batch_scores = softmax(batch_validation_scores)
            
            val_loss = nx.sum(cross_entropy(softmax_batch_scores, next_tokens), dtype=nx.float32)

            total_loss += val_loss.item()
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
    
   
    def predict(self, context:Any, embedding:Embedding) -> Any:
        embedded = embedding.forward(context)
        # embedded = embedded[None, :, :]
        scores = self.forward(embedded, embedding, False)
        return scores