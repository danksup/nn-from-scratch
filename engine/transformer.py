from engine.feedforward import Layer
from engine.losses import cross_entropy_gradient, cross_entropy
from engine.activations import softmax
from engine.embedding import Embedding
from engine.dataloader import DataLoader
from engine.optimizer import SGD, Momentum, Adam, AdamW
from engine.positionalencoding import PE
from engine.attention import AttentionLayer
from engine.transformer_block import TransformerBlock
import numpy as np
import time
import json

class Transformer:
    def __init__(self, vocab_size:int, embed_dim:int,optimizer=None) -> None:
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.blocks = []
        self.classifier = Layer.output(vocab_size, embed_dim)
        if optimizer is None:
            optimizer = Adam()

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
        
    def forward(self, inputs:np.ndarray) -> np.ndarray:
        '''
        inputs = list of inputs
        '''
        output = inputs
        for block in self.blocks:
            output = block.forward(output)
        self.last_token = output[:, -1,:]
        self.scores = self.classifier.forward(self.last_token)
        
        self.last_output = output
        return self.scores
    
    def backward(self, err_signal:np.ndarray) -> np.ndarray:
        '''
        Args:
            err_signal: gradient
        '''
        gradient = self.classifier.backward(err_signal)

        full_grad = np.zeros_like(self.last_output)
        full_grad[:, -1, :] = gradient
        cuurent_grad = full_grad

        for block in self.blocks[::-1]:
            err_signal = block.backward(cuurent_grad)
            cuurent_grad = err_signal
        
        for block in self.blocks:
            self.optimizer.step(block.attention.Wq, block.attention.dWq)
            self.optimizer.step(block.attention.Wk, block.attention.dWk)
            self.optimizer.step(block.attention.Wv, block.attention.dWv)
            self.optimizer.step(block.attention.Bq, block.attention.dBq)
            self.optimizer.step(block.attention.Bk, block.attention.dBk)
            self.optimizer.step(block.attention.Bv, block.attention.dBv)
            self.optimizer.step(block.ff1.weights, block.ff1.d_weight)
            self.optimizer.step(block.ff1.biases, block.ff1.d_bias)
            self.optimizer.step(block.ff2.weights, block.ff2.d_weight)
            self.optimizer.step(block.ff2.biases, block.ff2.d_bias)
        
        self.optimizer.step(self.classifier.weights,self.classifier.d_weight)
        self.optimizer.step(self.classifier.biases,self.classifier.d_bias)
                        
        return cuurent_grad

    def train(self, dataloader:DataLoader, embedding:Embedding, batch_size:int=32):
        '''
        Args:
            data: dataset, ex: [(1,1), (2,4), (3,9)] -> x^2
            epochs: one complete pass of training
            lr = learning rate, how big a step the model takes when adjusting weights.
            print_loss: output loss
        '''
        total_loss = 0.0
        count = 0

        for contexts, next_tokens in dataloader.get_pairs(batch_size):            
            embedded = embedding.lookup_table[contexts]  # shape (batch, context_size, embed_dim)
            embedded += PE(dataloader.context_size, embedding.embed_dim)
            batch_scores = self.forward(embedded)


            softmax_batch_scores = softmax(batch_scores)
            batch_gradient = cross_entropy_gradient(softmax_batch_scores, next_tokens)

            loss = np.sum(cross_entropy(softmax_batch_scores, next_tokens))
            total_loss += loss
            count += contexts.shape[0]
            error_signal = self.backward(batch_gradient)
         
            embedding_gradient = np.zeros_like(embedding.lookup_table)
            np.add.at(embedding_gradient, contexts, error_signal)
            self.optimizer.step(embedding.lookup_table, embedding_gradient)  
        return total_loss / count
    
    # def count_params_model(self):
    #     """
    #     count the model's param
    #     """
    #     total = 0
    #     for layer in self.layers:
    #         total += layer.weights.size
    #         total += layer.biases.size
    #     return total
    
    def to_dict(self) -> dict:
        transformer = {
            "vocab_size":self.vocab_size,
            "embed_dim": self.embed_dim,
            "blocks": [],
            "classifier": None
        }
        for block in self.blocks:
            transformer["blocks"].append(block.to_dict())
        transformer["classifier"] = self.classifier.to_dict()
        return transformer
    
    @classmethod
    def from_dict(cls,thing:dict):
        vocab_size = thing["vocab_size"]
        embed_dim = thing["embed_dim"]
        blocks = thing["blocks"]
        classifier = thing["classifier"]
        transformer = cls(vocab_size, embed_dim)
        for block in blocks:
            transformer.add_block(TransformerBlock.from_dict(block))
        transformer.classifier = Layer.from_dict(classifier)
        return transformer
    
   
    def predict(self, context:np.ndarray, embedding:Embedding) -> np.ndarray:
        embedded = embedding.forward(context) + PE(len(context), embedding.embed_dim)
        scores = self.forward(embedded)
        return scores