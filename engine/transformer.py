from engine.feedforward import Layer
from engine.losses import cross_entropy_gradient, cross_entropy
from engine.activations import softmax
from engine.embedding import Embedding
from engine.dataloader import DataLoader
from engine.optimizer import  AdamW
from engine.positional_encoding import PE
from engine.transformer_block import TransformerBlock
from engine.backend import nx
from typing import Any


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
        
    def forward(self, inputs:Any, embedding:Embedding) -> Any:
        '''
        inputs = list of inputs
        '''
        output = inputs
        for block in self.blocks:
            output = block.forward(output)
        self.last_output = output
        scores = output @ embedding.lookup_table.T
        return scores
    
    def backward(self, err_signal:Any, embedding_table:Any) -> Any:
        '''
        Args:
            traces error contribution and then optimize
        '''
        block_gradient = err_signal @ embedding_table
        d_table = err_signal.reshape(-1, self.vocab_size).T @ self.last_output.reshape(-1, self.embed_dim)
        current_grad = block_gradient
        for block in self.blocks[::-1]:
            current_grad = block.backward(current_grad)
        
        for i,block in enumerate(self.blocks):
            self.optimizer.step(f"Wq_{i}", block.attention.Wq, block.attention.dWq)
            self.optimizer.step(f"Wk_{i}",block.attention.Wk, block.attention.dWk)
            self.optimizer.step(f"Wv_{i}",block.attention.Wv, block.attention.dWv)
            self.optimizer.step(f"Wo_{i}",block.attention.Wo, block.attention.dWo)
            self.optimizer.step(f"ff1_weights_{i}",block.ff1.weights, block.ff1.d_weight)
            self.optimizer.step(f"ff1_biases_{i}",block.ff1.biases, block.ff1.d_bias)
            self.optimizer.step(f"ff2_weights_{i}",block.ff2.weights, block.ff2.d_weight)
            self.optimizer.step(f"ff2_biases_{i}",block.ff2.biases, block.ff2.d_bias)
            self.optimizer.step(f"rmsnorm1_gamma_{i}", block.rmsnorm1.gamma, block.rmsnorm1.d_gamma)
            self.optimizer.step(f"rmsnorm2_gamma_{i}", block.rmsnorm2.gamma, block.rmsnorm2.d_gamma)
        

        return current_grad,d_table

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
            embedded += PE(dataloader.context_size, embedding.embed_dim)
            batch_scores = self.forward(embedded, embedding)

            softmax_batch_scores = softmax(batch_scores)
            batch_gradient = cross_entropy_gradient(softmax_batch_scores, next_tokens)

            loss = nx.sum(cross_entropy(softmax_batch_scores, next_tokens), dtype=nx.float32)
            nx.eval(loss)
            total_loss += float(loss)
            count += next_tokens.size
            current_grad,d_table = self.backward(batch_gradient, embedding.lookup_table)
            embedding_gradient = nx.zeros_like(embedding.lookup_table)
            embedding_gradient = nx.add_at(embedding_gradient, contexts, current_grad)
            # print(embedding_gradient.shape)
            # print(nx.sum(nx.abs(embedding_gradient)))
            # print(
            #     contexts.shape,
            #     current_grad.shape,
            #     embedding_gradient.shape
            # )
            # print(
            #     nx.sum(nx.abs(embedding_gradient)),
            #     nx.sum(nx.abs(d_table))
            # )
            total_embedding_gradient = embedding_gradient + d_table
            self.optimizer.step("embedding",embedding.lookup_table, total_embedding_gradient)  
        return nx.float_32(total_loss / count)
    
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
        embedded = embedding.forward(context) + PE(len(context), embedding.embed_dim)
        scores = self.forward(embedded, embedding)
        return scores