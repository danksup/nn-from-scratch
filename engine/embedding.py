import random
import json
import engine.backend as nx
from typing import Any


class Embedding:
    def __init__(self, n:int, embed_dim:int) -> None:
        self.embed_dim = embed_dim
        self.lookup_table = nx.uniform(low=-0.1, high=0.1, size=(n, self.embed_dim), dtype=nx.float32)

    def forward(self, token_list:Any):
        ''' loopup and convert to the vector for each token id'''
        
        return self.lookup_table[token_list]
    
    def to_dict(self) -> dict:
        lookup = {"lookuptable": self.lookup_table.tolist()}
        return lookup

    @classmethod
    def from_dict(cls, thing:dict) -> "Embedding":
        lookuptable = nx.array(thing["lookuptable"], dtype=nx.float32)
        embedding = cls(lookuptable.shape[0],lookuptable.shape[1])
        embedding.lookup_table = lookuptable
        return embedding