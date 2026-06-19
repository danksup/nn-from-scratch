from engine.layer import Layer
from engine.losses import cross_entropy_gradient, cross_entropy
import json
from engine.activations import softmax
from engine.embedding import Embedding
from engine.dataloader import DataLoader
from engine.optimizer import SGD, Momentum, Adam
import numpy as np
import time

class Model:
    def __init__(self, optimizer=None) -> None:
        self.layers = []
        if optimizer is None:
            optimizer = Adam()

        self.optimizer = optimizer
    def __repr__(self) -> str:
        str_layers = ""
        for idx,i in enumerate(self.layers):
            str_layers += f"layer{idx}:{str(len(i))} neurons"  + "\n"
        str_layers += f"total layer: {len(self.layers)}, params: {self.count_params()}" 
        return str_layers
    
    def add_layer(self, layer:Layer) -> None:
        """
        add a layer object to the model.
        """
        self.layers.append(layer)
    
    def add_layers(self, layers:list[Layer]) -> None:
        '''
        add multiple layers to the model.
        '''
        self.layers.extend(layers)

    @classmethod
    def build(cls, input_size:int, output_size:int, hidden_layer_size:int=1, base_width:int=512) -> "Model":
        '''
        Args:
            input_size: input size
            output_size: output size
            hidden_layer_size: number or hidden layers
            base_width: number of neurons per hidden layer
        construct a brand new model
        '''
        count = 1
        model = cls()

        #hidden layer
        model.add_layer(Layer.hidden(base_width, input_size))

        for _ in range(count,hidden_layer_size):
            model.add_layer(Layer.hidden(base_width,base_width))

        #output layer
        model.add_layer(Layer.output(output_size, base_width))
        return model

    def forward(self, inputs:np.ndarray) -> np.ndarray:
        '''
        inputs = list of inputs
        '''
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        
        self.last_output = output
        return output
    
    def backward(self, err_signal:np.ndarray) -> np.ndarray:
        '''
        Args:
            err_signal: gradient
            lr: learning rate, how big a step the model takes when adjusting weights.
        '''
        for layer in self.layers[::-1]:
            err_signal = layer.backward(err_signal)
        
        for layer in self.layers:
            self.optimizer.step(layer.weights, layer.d_weight)
            self.optimizer.step(layer.biases, layer.d_bias)
        
        return err_signal

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

        # forward_time = 0
        # backward_time = 0
        # embedding_update = 0
        for batch in dataloader.get_pairs(batch_size):
            contexts = [pair[0] for pair in batch]
            next_tokens = [pair[1] for pair in batch]
            
            embedded = embedding.lookup_table[contexts]  # shape (batch, context_size, embed_dim)
            flat = embedded.reshape(len(contexts), -1)        
            # start = time.perf_counter()     
            batch_scores = self.forward(flat)
            # end = time.perf_counter()
            # forward_time += end-start

            softmax_batch_scores = softmax(batch_scores)
            batch_gradient = np.array(cross_entropy_gradient(softmax_batch_scores, next_tokens)) 

            loss = np.sum(cross_entropy(softmax_batch_scores, next_tokens))
            total_loss += loss
            count += len(batch)

            # start = time.perf_counter()
            error_signal = self.backward(batch_gradient).reshape(len(contexts),dataloader.context_size,embedding.embed_dim)
            # end = time.perf_counter()
            # backward_time += end-start
            # start = time.perf_counter()
            embedding_gradient = np.zeros_like(embedding.lookup_table)
            np.add.at(embedding_gradient, contexts, error_signal)
            self.optimizer.step(embedding.lookup_table, embedding_gradient)
            # end = time.perf_counter()
            # embedding_update += end-start

        # print(f"backward_time: {backward_time:.5f} | embedding_update: {embedding_update:.5f} | forward_time: {forward_time:.5f}")   
        return total_loss / count
    
    def count_params(self) -> int:
        """
        count how many parameters this model has.
        """
        total_params = 0
        for layer in self.layers:
            total_params += layer.n * layer.m  + len(layer.biases)
        
        return total_params
    
    def to_dict(self) -> dict:
        model = {
            "layers": []
        }
        for layer in self.layers:
            model["layers"].append(
                 {
                    "n":layer.n,
                    "m":layer.m,
                    "weights":layer.weights.tolist(),
                    "biases":layer.biases.tolist(),
                }
            )
        return model
    
    def save(self, filename:str):
        """
        save model into a JSON file.
        """
        model = self.to_dict()
        filename = f'artifacts/models/model_{str(self.count_params())}_{filename}.json'
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(model, file, indent=4)

    @classmethod
    def from_dict(cls,thing:dict):
        model = cls()
        for layer in thing["layers"]:
            current = Layer(layer["n"], layer["m"])
            current.weights = np.array(layer["weights"])
            current.biases = np.array(layer["biases"])
            model.add_layer(current)

        return model
    
    def load(self, filename:str) -> "Model":
        """
        load model from a JSON file
        """
        try:
            with open(filename, "r") as file:
                model = json.load(file)

            self.layers = Model.from_dict(model).layers
            return self

        except FileNotFoundError:
            raise FileNotFoundError("file not found")
        except json.JSONDecodeError:
            raise ValueError("decode eror")

    def predict(self, context:np.ndarray, embedding:Embedding) -> np.ndarray:
        embedded = embedding.forward(context)
        flat = embedded.flatten()
        scores = self.forward(flat)
        return scores