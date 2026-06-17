import math

#rectified linear unit
def relu(x) ->int:
    return max(0,x)

def relu_derivative(x) -> int:
    return 0 if x < 0 else 1 #irl, x is unlikely to be 0 so we can ignore it

def leaky_relu(x) -> float:
    return x if x > 0 else 0.01 * x

def leaky_relu_derivative(x) ->float:
    return 1 if x > 0 else 0.01

def softmax(l:list[float]) -> list[float]:
    max_l = max(l)
    exponent = [math.exp(i - max_l) for i in l] #if logit becomes too high, it can explode
    return [i/sum(exponent) for i in exponent]