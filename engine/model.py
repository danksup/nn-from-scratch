from engine.layer import Layer
import json

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
    
    def backward(self, actual:list, lr:float=1e-3):
        '''
        actual = list of actual answer from database
        lr = learning rate, how big a step the model takes when adjusting weights.
        '''
        predictions = self.last_output
        err_signal = []
        for i, prediction in enumerate(predictions):
            initial_error = 2 * (prediction - actual[i])
            err_signal.append(initial_error)

        for layer in self.layers[::-1]:
            err_signal = layer.backward(err_signal, lr)

    def train(self, data:list, epochs:int, lr:float=1e-3, print_loss:bool=False):
        '''
        data: dataset, ex: [(1,1), (2,4), (3,9)] -> x^2
        epochs: one complete pass of training
        lr = learning rate, how big a step the model takes when adjusting weights.
        print_loss: output loss
        '''
        for i in range(epochs):
            for x,actual in data:
                self.forward([x])
                self.backward([actual], lr)
        
            if print_loss and i % 100 == 0:
                for x, actual in data:
                    pred = self.forward([x])
                    print(f"x={x} | pred={pred[0]:.2f} | actual={actual}")
                print("---")
    
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
            raise FileNotFoundError("kys")
        except json.JSONDecodeError:
            raise ValueError("ks")

    def predict(self, data):
        return self.forward(data)


        

    

        