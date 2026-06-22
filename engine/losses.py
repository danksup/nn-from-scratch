import numpy as np

def cross_entropy(probs:np.ndarray, target_idx):
    '''
    |  |I

    ||   |_
    '''
    if probs.ndim == 2:
        row_coords = np.arange(probs.shape[0])
        p = np.clip(probs[row_coords, target_idx], np.float32(1e-12), np.float32(1.0), dtype=np.float32)
        return -np.log(p, dtype=np.float32)
    elif probs.ndim == 3:

        B, T, V = probs.shape

        row_b, row_t = np.indices((B, T))

        p = probs[row_b, row_t, target_idx]

        p = np.clip(p, 1e-12, 1.0)

        return -np.log(p)
    else:
        return -np.log(probs[target_idx], dtype=np.float32)

def cross_entropy_gradient(probs:np.ndarray , target_idx) -> np.ndarray:
    '''
    derivative of cross entropy and softmax
    '''
    probs_copy = probs.copy()
    if probs.ndim == 2:
        row_coords = np.arange(probs.shape[0])
        probs_copy[row_coords, target_idx] -= np.float32(1.0)
    elif probs.ndim == 3:

        B, T, _ = probs.shape

        row_b, row_t = np.indices((B, T))

        probs_copy[row_b, row_t, target_idx] -= 1.0
    else:
        probs_copy[target_idx] = probs_copy[target_idx]  - np.float32(1)
    return probs_copy
