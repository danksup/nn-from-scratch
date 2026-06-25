import random
import json
from engine.backend import nx
from typing import Any


class Embedding:
    def __init__(self, n:int, embed_dim:int) -> None:
        self.embed_dim = embed_dim
        self.lookup_table = nx.uniform(low=-0.1, high=0.1, size=(n, self.embed_dim), dtype=nx.float16)
        self.f32_embedding_lookuptable = self.lookup_table.astype(nx.float32)

    def forward(self, token_list:Any):
        ''' loopup and convert to the vector for each token id'''

        return self.lookup_table[token_list]
    
    def to_dict(self) -> dict:
        lookup = {"lookuptable": self.lookup_table.tolist()}
        return lookup

    def save(self, filename:str):
        '''
        save the embedding
        '''
        lookup = self.to_dict()
        filename = "embedding_" + filename
        with open(f"artifacts/embeds/{filename}.json", "w") as f:
            json.dump(lookup, f, indent=4)

    @classmethod
    def from_dict(cls, thing:dict) -> "Embedding":
        lookuptable = nx.array(thing["lookuptable"], dtype=nx.float32)
        embedding = cls(lookuptable.shape[0],lookuptable.shape[1])
        embedding.lookup_table = lookuptable
        return embedding


    @classmethod
    def load(cls, filename:str) -> "Embedding":
        """
        load the embedding
        """
        try:
            with open(filename, 'r') as f:
                loaded = json.load(f)["lookuptable"]

            embedding = cls(len(loaded), len(loaded[0]))
            embedding.lookup_table = nx.array(loaded, dtype=nx.float16)
            return embedding
        except FileNotFoundError:
            raise FileNotFoundError("file not found")
        except json.JSONDecodeError:
            raise ValueError("decode eror")
