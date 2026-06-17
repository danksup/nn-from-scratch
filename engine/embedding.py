import random

class Embedding:
    def __init__(self, n:int, embed_dim:int) -> None:
        self.lookup_table = [[random.uniform(-0.1, 0.1) for _ in range(embed_dim)] for _ in range(n)]
    
    def forward(self, token_list:list[int]):
        ''' loopup and convert to the vector for each token id'''
        print(token_list)
        print(self.lookup_table)
        return [self.lookup_table[i] for i in token_list]
