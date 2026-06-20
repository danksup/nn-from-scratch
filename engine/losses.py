import numpy as np

def cross_entropy(probs:np.ndarray, target_idx):
    '''
    |  |I

    ||   |_
    '''
    if probs.ndim == 2:
        row_coords = np.arange(probs.shape[0])
        p = np.clip(probs[row_coords, target_idx], 1e-12, 1.0)
        return -np.log(p)
    else:
        return -np.log(probs[target_idx])

def cross_entropy_gradient(probs:np.ndarray , target_idx) -> np.ndarray:
    '''
    derivative of cross entropy and softmax
    '''
    probs_copy = probs.copy()
    if probs.ndim == 2:
        row_coords = np.arange(probs.shape[0])
        probs_copy[row_coords, target_idx] -= 1
    else:
        probs_copy[target_idx] = probs_copy[target_idx]  - 1
    return probs_copy
