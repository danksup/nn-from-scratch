from engine.layer import Layer

class Model:
    def __init__(self) -> None:
        self.layers = []

    def __repr__(self) -> str:
        str_layers = ""
        for i in self.layers:
            str_layers += i + "\n" 
        return str_layers
    
    def add_layer(self, layer:Layer) -> None:
        self.layers.append(layer)

    def forward(self, inputs:list):
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        
        self.last_output = output
        return output
    
    def backward(self, actual:list, lr:float=1e-3):
        predictions = self.last_output
        err_signal = []
        for i, prediction in enumerate(predictions):
            initial_error = 2 * (prediction - actual[i])
            err_signal.append(initial_error)

        for layer in self.layers[::-1]:
            err_signal = layer.backward(err_signal, lr)

    def train(self, data:list, epochs:int, lr:float=1e-3, print_loss:bool=False):
        res = []
        for i in range(epochs):
            for x,actual in data:
                self.forward([x])
                self.backward([actual], lr)
        
            if print_loss and i % 100 == 0:
                for x, actual in data:
                    pred = self.forward([x])
                    print(f"x={x} | pred={pred[0]:.2f} | actual={actual}")
                print("---")
    
    def count_params(self):
        total_params = 0
        for layer in self.layers:
            total_params += layer.n * layer.m  + len(layer.biases)
        
        return total_params
      

        