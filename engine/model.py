from engine.layer import Layer
from engine.losses import cross_entropy_gradient, cross_entropy
import json
from engine.activations import softmax
from engine.embedding import Embedding
from engine.dataloader import DataLoader
import numpy as np

class Model:
    def __init__(self) -> None:
        self.layers = []

    def __repr__(self) -> str:
        str_layers = ""
        for i in self.layers:
            str_layers += str(i) + "\n" 
        return str_layers
    
    def add_layer(self, layer:Layer) -> None:
        """
        add layer object.
        """
        self.layers.append(layer)

    def forward(self, inputs:np.ndarray) -> np.ndarray:
        '''
        inputs = list of inputs
        '''
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        
        self.last_output = output
        return output
    
    def backward(self, err_signal:np.ndarray, lr:float=1e-3):
        '''
        Args:
            err_signal: gradient
            lr: learning rate, how big a step the model takes when adjusting weights.
        '''
        for layer in self.layers[::-1]:
            err_signal = layer.backward(err_signal, lr)

    def train(self, dataloader:DataLoader, embedding:Embedding, epochs:int, lr:float=1e-3, print_loss:bool=False, batch_size:int=32):
        '''
        Args:
            data: dataset, ex: [(1,1), (2,4), (3,9)] -> x^2
            epochs: one complete pass of training
            lr = learning rate, how big a step the model takes when adjusting weights.
            print_loss: output loss
        '''
        for i in range(epochs):
            total_loss = 0.0

            count = 0
            for batch in dataloader.get_pairs(batch_size):
                contexts = [pair[0] for pair in batch]
                next_tokens = [pair[1] for pair in batch]
                
                embedded = embedding.lookup_table[contexts]  # shape (batch, context_size, embed_dim)
                flat = embedded.reshape(len(contexts), -1)             
                batch_scores = self.forward(flat)
                batch_gradient = np.array(cross_entropy_gradient(softmax(batch_scores), next_tokens)) 

                loss = np.sum(cross_entropy(softmax(batch_scores), next_tokens))
                total_loss += loss
                count += len(batch)

                self.backward(batch_gradient, lr)
        
            if print_loss:
                print(f"epoch={i} | avg_loss={total_loss / count:.4f}")
    
    def count_params(self) -> int:
        """
        count how many parameters this model has.
        """
        total_params = 0
        for layer in self.layers:
            total_params += layer.n * layer.m  + len(layer.biases)
        
        return total_params
    
    def save(self, filename:str):
        """
        save model into a JSON file.
        """
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
        filename = f'artifacts/models/model_{str(self.count_params())}_{filename}.json'
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(model, file, indent=4)
    
    def load(self, filename:str) -> "Model":
        """
        load model from a JSON file
        """
        try:
            with open(filename, "r") as file:
                model = json.load(file)

            self.layers = []
            for layer in model["layers"]:
                current = Layer(layer["n"], layer["m"])
                current.weights = np.array(layer["weights"])
                current.biases = np.array(layer["biases"])
                self.add_layer(current)
            
            return self

        except FileNotFoundError:
            raise FileNotFoundError("file not found")
        except json.JSONDecodeError:
            raise ValueError("decode eror")

    def predict(self, context:np.ndarray, embedding:Embedding) -> tuple[np.ndarray, int]:
        embedded = embedding.forward(context)
        flat = embedded.flatten()
        scores = self.forward(flat)
        probs = softmax(scores)
        return probs, int(probs.argmax())


        

    

        