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
        self.classifier = Layer.output(vocab_size, embed_dim)
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
        
    def forward(self, inputs:Any) -> Any:
        '''
        inputs = list of inputs
        '''
        output = inputs
        for block in self.blocks:
            output = block.forward(output)
        scores = self.classifier.forward(output)
        return scores
    
    def backward(self, err_signal:Any) -> Any:
        '''
        Args:
            traces error contribution and then optimize
        '''
        gradient = self.classifier.backward(err_signal)
        cuurent_grad = gradient

        for block in self.blocks[::-1]:
            err_signal = block.backward(cuurent_grad)
            cuurent_grad = err_signal
        
        for i,block in enumerate(self.blocks):
            self.optimizer.step(f"Wq_{i}", block.attention.Wq, block.attention.dWq)
            self.optimizer.step(f"Wk_{i}",block.attention.Wk, block.attention.dWk)
            self.optimizer.step(f"Wv_{i}",block.attention.Wv, block.attention.dWv)
            self.optimizer.step(f"Bq_{i}",block.attention.Bq, block.attention.dBq)
            self.optimizer.step(f"Bk_{i}",block.attention.Bk, block.attention.dBk)
            self.optimizer.step(f"Bv_{i}",block.attention.Bv, block.attention.dBv)
            self.optimizer.step(f"ff1_weights_{i}",block.ff1.weights, block.ff1.d_weight)
            self.optimizer.step(f"ff1_biases_{i}",block.ff1.biases, block.ff1.d_bias)
            self.optimizer.step(f"ff2_weights_{i}",block.ff2.weights, block.ff2.d_weight)
            self.optimizer.step(f"ff2_biases_{i}",block.ff2.biases, block.ff2.d_bias)
            self.optimizer.step(f"rmsnorm1_gamma_{i}", block.rmsnorm1.gamma, block.rmsnorm1.d_gamma)
            self.optimizer.step(f"rmsnorm2_gamma_{i}", block.rmsnorm2.gamma, block.rmsnorm2.d_gamma)
        
        self.optimizer.step("classifier_weights",self.classifier.weights,self.classifier.d_weight)
        self.optimizer.step("classifier_biases",self.classifier.biases,self.classifier.d_bias)
        return cuurent_grad

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
            batch_scores = self.forward(embedded)

            softmax_batch_scores = softmax(batch_scores)
            batch_gradient = cross_entropy_gradient(softmax_batch_scores, next_tokens)

            loss = nx.sum(cross_entropy(softmax_batch_scores, next_tokens), dtype=nx.float32)
            nx.eval(loss)
            total_loss += float(loss)
            count += next_tokens.size
            error_signal = self.backward(batch_gradient)
         
            embedding_gradient = nx.zeros_like(embedding.lookup_table)
            embedding_gradient = nx.add_at(embedding_gradient, contexts, error_signal)
            self.optimizer.step("embedding",embedding.lookup_table, embedding_gradient)  
        return nx.float_32(total_loss / count)
    
    def to_dict(self) -> dict:
        """
        get dictionary
        """
        transformer = {
            "vocab_size":self.vocab_size,
            "embed_dim": self.embed_dim,
            "blocks": [],
            "classifier": None,
        }
        for block in self.blocks:
            transformer["blocks"].append(block.to_dict())
        transformer["classifier"] = self.classifier.to_dict()
        return transformer
    
    @classmethod
    def from_dict(cls,thing:dict) -> "Transformer":
        """init from dictionary"""
        vocab_size = thing["vocab_size"]
        embed_dim = thing["embed_dim"]
        blocks = thing["blocks"]
        classifier = thing["classifier"]
        transformer = cls(vocab_size, embed_dim)
        for block in blocks:
            transformer.add_block(TransformerBlock.from_dict(block))
        transformer.classifier = Layer.from_dict(classifier)
        return transformer
    
   
    def predict(self, context:Any, embedding:Embedding) -> Any:
        embedded = embedding.forward(context) + PE(len(context), embedding.embed_dim)
        scores = self.forward(embedded)
        return scores