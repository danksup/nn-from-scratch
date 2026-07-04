from engine.embedding import Embedding
from engine.tokenizer import Tokenizer
import engine.backend as nx 

def n_closest(word:str, tokenizer:Tokenizer ,embedding:Embedding, n:int=10):
    lookuptable = embedding.lookup_table
    encoded = tokenizer.encode(word)

    encoded_embedding = embedding.forward(encoded)
    whole_embedding = nx.mean(encoded_embedding, axis=0)

    query_norm = nx.norm(whole_embedding,ord=2, keepdims=True)
    table_norms = nx.norm(lookuptable, ord=2, axis=1, keepdims=False)
    distance = (whole_embedding @ lookuptable.T) / (query_norm * table_norms) #type:ignore
    ids = nx.argsort(distance)[::-1]
    top_ids = ids[:n]

    decoded = []
    for i in top_ids:
        print(tokenizer.decode([i.item()]))
    