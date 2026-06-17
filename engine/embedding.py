import random
import json

class Embedding:
    def __init__(self, n:int, embed_dim:int) -> None:
        self.lookup_table = [[random.uniform(-0.1, 0.1) for _ in range(embed_dim)] for _ in range(n)]
    
    def forward(self, token_list:list[int]):
        ''' loopup and convert to the vector for each token id'''
        return [self.lookup_table[i] for i in token_list]

    def save(self, filename:str):
        '''
        save the embedding
        '''
        lookup = {"lookuptable": self.lookup_table}
        with open(f"embeds/{filename}.json", "w") as f:
            json.dump(lookup, f, indent=4)

    def load(self, filename:str):
        """
        load the embedding
        """
        try:
            self.lookup_table = []
            with open(filename, 'r') as f:
                self.lookup_table = json.load(f)["lookuptable"]

            return self
        except FileNotFoundError:
            raise FileNotFoundError("file not found")
        except json.JSONDecodeError:
            raise ValueError("decode eror")
