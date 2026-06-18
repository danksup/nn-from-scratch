import numpy as np

def cross_entropy(probs:np.ndarray, target_idx):
    '''
    |  |I

    ||   |_
    '''
    if probs.ndim == 2:
        loss = np.ndarray(len(probs))
        for i in range(len(probs)):
            loss[i] = -np.log(probs[i,target_idx[i]])
        
        return loss
    else:
        return -np.log(probs[target_idx])

def cross_entropy_gradient(probs:np.ndarray , target_idx) -> np.ndarray:
    '''
    derivative of cross entropy and softmax
    '''
    probs_copy = probs.copy()

    if probs.ndim == 2:
        for i in range(len(probs_copy)):
            probs_copy[i, target_idx[i]] = probs_copy[i, target_idx[i]] -1
   
    else:
        probs_copy[target_idx] = probs_copy[target_idx]  - 1
            
    
    return probs_copy
