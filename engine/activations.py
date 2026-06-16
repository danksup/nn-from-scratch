#rectified linear unit
def relu(x):
    return max(0,x)

def relu_derivative(x):
    return 0 if x < 0 else 1 #irl, x is unlikely to be 0 so we can ignore it

def leaky_relu(x):
    return x if x > 0 else 0.01 * x

def leaky_relu_derivative(x):
    return 1 if x > 0 else 0.01