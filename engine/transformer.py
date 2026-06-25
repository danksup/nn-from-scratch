from engine.feedforward import Layer
from engine.losses import cross_entropy_gradient, cross_entropy
from engine.activations import softmax
from engine.embedding import Embedding
from engine.dataloader import DataLoader
from engine.optimizer import  AdamW
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
        output = inputs.astype(nx.float16)
        for block in self.blocks:
            output = block.forward(output)
        self.last_output =output.astype(nx.float32)
        # print("last output shape",self.last_output.shape)
        # print("f32 lookuptable shape",embedding.f32_embedding_lookuptable.shape)
        # print("f32 lookuptable transposed shape",embedding.f32_embedding_lookuptable.transpose(1,0).shape)
        scores = self.last_output @ embedding.f32_embedding_lookuptable.T
        return scores
    
    def backward(self, err_signal:Any, embedding:Embedding) -> Any:
        '''
        Args:
            traces error contribution and then optimize
        '''
        err_signal =  err_signal.astype(nx.float32)
        block_gradient = err_signal @ embedding.f32_embedding_lookuptable
        d_table = err_signal.reshape(-1, self.vocab_size).T @ self.last_output.reshape(-1, self.embed_dim)
        d_table /= (err_signal.shape[0]* err_signal.shape[1])
        
        current_grad = block_gradient
        for block in self.blocks[::-1]:
            current_grad = block.backward(current_grad)
        
        for i,block in enumerate(self.blocks):
            optimized = self.optimizer.step_many(
                ((f"Wq_{i}", block.attention.Wq, block.attention.dWq),
                (f"Wk_{i}", block.attention.Wk, block.attention.dWk),
                (f"Wv_{i}", block.attention.Wv, block.attention.dWv),
                (f"Wo_{i}", block.attention.Wo, block.attention.dWo),
                (f"ff1_weights_{i}", block.ff1.weights, block.ff1.d_weight),
                (f"ff1_biases_{i}", block.ff1.biases, block.ff1.d_bias),
                (f"ff2_weights_{i}", block.ff2.weights, block.ff2.d_weight),
                (f"ff2_biases_{i}", block.ff2.biases, block.ff2.d_bias),
                (f"rmsnorm1_gamma_{i}", block.rmsnorm1.gamma, block.rmsnorm1.d_gamma),
                (f"rmsnorm2_gamma_{i}", block.rmsnorm2.gamma, block.rmsnorm2.d_gamma))
            )
            block.attention.Wq = optimized[f"Wq_{i}"]
            block.attention.Wk = optimized[f"Wk_{i}"]
            block.attention.Wv = optimized[f"Wv_{i}"]
            block.attention.Wo = optimized[f"Wo_{i}"]
            block.ff1.weights = optimized[f"ff1_weights_{i}"]
            block.ff1.biases = optimized[f"ff1_biases_{i}"]
            block.ff2.weights = optimized[f"ff2_weights_{i}"]
            block.ff2.biases = optimized[f"ff2_biases_{i}"]
            block.rmsnorm1.gamma = optimized[f"rmsnorm1_gamma_{i}"]
            block.rmsnorm2.gamma = optimized[f"rmsnorm2_gamma_{i}"]
           
       
      
        return current_grad,d_table

    def train_mode(self):
        for block in self.blocks:
            block.train()

    def eval_mode(self):
        for block in self.blocks:
            block.eval()

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
            # embedded += PE(dataloader.context_size, embedding.embed_dim)
            
            batch_scores = self.forward(embedded, embedding)

            softmax_batch_scores = softmax(batch_scores)
            batch_gradient = cross_entropy_gradient(softmax_batch_scores, next_tokens)
            
            loss = nx.sum(cross_entropy(softmax_batch_scores, next_tokens), dtype=nx.float32)
            total_loss += loss
            count += next_tokens.size
            current_grad,d_table = self.backward(batch_gradient, embedding)
            embedding_gradient = nx.zeros_like(embedding.lookup_table, dtype=nx.float32)
            embedding_gradient = nx.add_at(embedding_gradient, contexts, current_grad)

            # print(embedding_gradient.shape)
            # print(nx.sum(nx.abs(embedding_gradient)))
            # print(
            #     contexts.shape,
            #     current_grad.shape,
            #     embedding_gradient.shape
            # )
            # print(f"sum abs embed grad {nx.sum(nx.abs(embedding_gradient))},sum abs dtable{nx.sum(nx.abs(d_table))}")
            total_embedding_gradient = embedding_gradient + d_table
            # print("tot embed grad shape",total_embedding_gradient.shape)
            # print("tot embed grad sum abs",nx.sum(nx.abs(total_embedding_gradient)))
            optimized = self.optimizer.step(("embedding",embedding.lookup_table, total_embedding_gradient))  
            embedding.lookup_table = optimized
            embedding.f32_embedding_lookuptable = embedding.lookup_table.astype(nx.float32)
            nx.eval(loss, embedding.lookup_table, embedding.f32_embedding_lookuptable)
        #     nx.eval(
        #     *[
        #         x
        #         for block in self.blocks
        #         for x in (
        #             block.attention.Wq,
        #             block.attention.Wk,
        #             block.attention.Wv,
        #             block.attention.Wo,
        #             block.ff1.weights,
        #             block.ff1.biases,
        #             block.ff2.weights,
        #             block.ff2.biases,
        #             block.rmsnorm1.gamma,
        #             block.rmsnorm2.gamma,
        #             )
        #         ]
        #     )
        # nx.eval(total_loss)
        return nx.float_32(total_loss / count)
    
    def validate(self, embedding,dataloader:DataLoader, batch_size, train_split=.9):
        total_loss = nx.float_32(0.0)
        count = 0
        self.eval_mode()
        dataloader.train_split = train_split
        for contexts, next_tokens in dataloader.get_validation_pairs(batch_size):
            embedded = embedding.forward(contexts) 
            batch_validation_scores = self.forward(embedded, embedding)
            softmax_batch_scores = softmax(batch_validation_scores)
            
            val_loss = nx.sum(cross_entropy(softmax_batch_scores, next_tokens), dtype=nx.float32)

            total_loss +=  val_loss
            count += next_tokens.size

        nx.eval(total_loss)
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
        embedded = embedding.forward(context)
        # embedded = embedded[None, :, :]
        scores = self.forward(embedded, embedding)
        return scores