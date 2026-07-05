from engine.embedding import Embedding
from engine.tokenizer import Tokenizer
import engine.backend as nx 


def embedding_of(word, tokenizer:Tokenizer ,embedding:Embedding):
    encoded = tokenizer.encode(word)
    encoded_embedding = embedding.forward(encoded)
    whole_embedding = nx.mean(encoded_embedding, axis=0)
    return whole_embedding

def n_closest(to:str | nx.ArrayLike, tokenizer:Tokenizer ,embedding:Embedding, n:int=10):
    lookuptable = embedding.lookup_table
    
    whole_embedding = to

    if isinstance(to, str):
        whole_embedding = embedding_of(to, tokenizer, embedding)

    query_norm = nx.norm(whole_embedding, ord=2, keepdims=True)  #type:ignore
    table_norms = nx.norm(lookuptable, ord=2, axis=1, keepdims=False)
    distance = (whole_embedding @ lookuptable.T) / (query_norm * table_norms) #type:ignore
    ids = nx.argsort(distance)[::-1]
    top_ids = ids[:n]
 
    for id in top_ids:
        print(tokenizer.decode([id.item()]), distance[id].item())


    