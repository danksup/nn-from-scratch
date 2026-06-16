from .activations import relu, relu_derivative, leaky_relu, leaky_relu_derivative
from .utils import dot


#1 neuron
def forward(inputs:list, weights:list, b:float, active:bool=True, activation = leaky_relu) -> float:
    '''
    inputs: list of inputs
    weights: list of weights
    b: bias

    output: 0 or 1
    '''
    return activation(dot(inputs, weights) + b) if active else (dot(inputs,weights) + b)

def forward_pass(inputs:list, all_weights:list, biases:list) -> list:
    res = []
    for idx, weights in enumerate(all_weights):
        res.append(forward(inputs, weights, biases[idx]))
    
    return res

#deprecated
def backward_pass(inputs:list, hidden_outputs:list, prediction:float, actual:float, output_weights:list, hidden_weights:list, output_biases:list, hidden_biases:list, lr=0.01, activation=leaky_relu_derivative):
    '''
    inputs = list of inputs
    hidden_outputs = output of each layer before actual output
    prediction: final output 
    actual: real answer in dataset
    weights, all weights: current weight
    biases2, all biases: current biases
    lr: learning rate
    '''
    raise DeprecationWarning("no")
    output_error = 2 * (prediction - actual) 

    #update output layers
    d_bias2 = 0
    for idx, hidden_output in enumerate(hidden_outputs):
        d_weights2 = output_error * hidden_output
        d_bias2 = output_error
        output_weights[0][idx] = output_weights[0][idx] - lr * d_weights2
    output_biases[0] = output_biases[0] - lr * d_bias2

    #update hidden layers
    for idx, weight2 in enumerate(hidden_weights):
        error_hidden = output_error *  output_weights[0][idx] * activation(hidden_outputs[idx])
        for ix, x in enumerate(inputs):
            d_weights = error_hidden * x
            d_bias = error_hidden

            hidden_weights[idx][ix] = hidden_weights[idx][ix] - lr * d_weights
            hidden_biases[idx] = hidden_biases[idx] - lr * d_bias 

            
    return hidden_weights, hidden_biases, output_weights, output_biases