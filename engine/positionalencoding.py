import numpy as np

def PE(context_size:int, embed_dim:int):
    positions = np.arange(context_size).reshape(-1,1)
    dims = np.arange(0,embed_dim,2)
    div_term = 10000 ** (dims / embed_dim)

    pe = np.zeros((context_size, embed_dim))
    pe[:, 0::2] = np.sin(positions / div_term)  
    pe[:, 1::2] = np.cos(positions / div_term)

    return pe