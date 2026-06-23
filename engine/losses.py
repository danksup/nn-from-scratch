from engine.backend import Backend
from typing import Any

nx = Backend()
def cross_entropy(probs:Any, target_idx):
    '''
    |  |I

    ||   |_
    '''
    if probs.ndim == 2:
        row_coords = nx.arange(probs.shape[0])
        p = nx.clip(probs[row_coords, target_idx], nx.float_32(1e-12), nx.float_32(1.0), dtype=nx.float32)
        return -nx.log(p, dtype=nx.float32)
    elif probs.ndim == 3:
        B, T, V = probs.shape
        row_b, row_t = nx.indices((B, T))
        p = probs[row_b, row_t, target_idx]
        p = nx.clip(p, 1e-12, 1.0)
        return -nx.log(p)
    else:
        return -nx.log(probs[target_idx], dtype=nx.float32)

def cross_entropy_gradient(probs:Any , target_idx) -> Any:
    '''
    derivative of cross entropy and softmax
    '''
    probs_copy = nx.copy(probs)
    if probs.ndim == 2:
        row_coords = nx.arange(probs.shape[0])
        probs_copy[row_coords, target_idx] -= nx.float_32(1.0)
    elif probs.ndim == 3:
        B, T, _ = probs.shape
        row_b, row_t = nx.indices((B, T))
        probs_copy[row_b, row_t, target_idx] -= 1.0
    else:
        probs_copy[target_idx] = probs_copy[target_idx]  - nx.float_32(1)
    return probs_copy
