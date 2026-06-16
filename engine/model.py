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

    def forward(self, inputs):
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        return output

    