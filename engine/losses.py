import math

def cross_entropy(probs, target_idx):

    return -math.log(probs[target_idx])

def cross_entropy_gradient(probs:list , target_idx:int):
    '''
    derivative of cross entropy and softmax
    '''
    probs_copy = probs.copy()
    for i in range(len(probs)):
        if i == target_idx:
            probs_copy[i] = probs[i] - 1
    
    return probs_copy
        
