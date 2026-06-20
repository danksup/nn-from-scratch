import random
import json
import numpy as np

class Embedding:
    def __init__(self, n:int, embed_dim:int) -> None:
        self.embed_dim = embed_dim
        rng = np.random.default_rng()
        self.lookup_table = rng.uniform(low=-0.1, high=0.1,size=(n, self.embed_dim))
    
    def forward(self, token_list:np.ndarray):
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
        lookuptable = thing["lookuptable"]
        embedding = cls(1,1)
        embedding.lookup_table = np.array(lookuptable)
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
            embedding.lookup_table = np.array(loaded)
            return embedding
        except FileNotFoundError:
            raise FileNotFoundError("file not found")
        except json.JSONDecodeError:
            raise ValueError("decode eror")
