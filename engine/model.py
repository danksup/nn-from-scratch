from engine.layer import Layer
from engine.losses import cross_entropy_gradient, cross_entropy
import json
from engine.activations import softmax
from engine.embedding import Embedding
from engine.dataloader import DataLoader

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

    def forward(self, inputs:list) -> list:
        '''
        inputs = list of inputs
        '''
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        
        self.last_output = output
        return output
    
    def backward(self, err_signal:list, lr:float=1e-3):
        '''
        Args:
            err_signal: gradient
            lr: learning rate, how big a step the model takes when adjusting weights.
        '''
        for layer in self.layers[::-1]:
            err_signal = layer.backward(err_signal, lr)

    def train(self, dataloader:DataLoader, embeding:Embedding, epochs:int, lr:float=1e-3, print_loss:bool=False):
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
            for context, next_token in dataloader.get_pairs():
                embedded = embeding.forward(context)
                flat = [val for vec in embedded for val in vec]
                scores = self.forward(flat)
                probs = softmax(scores)

                loss = cross_entropy(probs, next_token)
                total_loss += loss
                count += 1

                err_signal = cross_entropy_gradient(probs, next_token)
                self.backward(err_signal, lr)
        
            if print_loss and i % 100 == 0:
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
                    "weights":layer.weights,
                    "biases":layer.biases,
                }
            )
        filename = f'models/model_{str(self.count_params())}_{filename}.json'
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
                current.weights = layer["weights"]
                current.biases = layer["biases"]
                self.add_layer(current)
            
            return self

        except FileNotFoundError:
            raise FileNotFoundError("file not found")
        except json.JSONDecodeError:
            raise ValueError("decode eror")

    def predict(self, context, embedding):
        embedded = embedding.forward(context)
        flat = [val for vec in embedded for val in vec]
        scores = self.forward(flat)
        probs = softmax(scores)
        return probs.index(max(probs))


        

    

        