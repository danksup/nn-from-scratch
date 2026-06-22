import numpy as np

def PE(context_size:int, embed_dim:int):
    positions = np.arange(context_size,dtype=np.float32).reshape(-1,1)
    dims = np.arange(0,embed_dim,2, dtype=np.float32)
    div_term = np.power(np.float32(10000.0),np.float32(dims / embed_dim), dtype=np.float32)

    pe = np.zeros((context_size, embed_dim),dtype=np.float32)
    pe[:, 0::2] = np.sin(positions / div_term, dtype=np.float32)  
    pe[:, 1::2] = np.cos(positions / div_term,dtype=np.float32)

    return pe