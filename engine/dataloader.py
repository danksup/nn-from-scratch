from engine.tokenizer import Tokenizer
import numpy as np
class DataLoader:
    def __init__(self,filepath:str, tokenizer:Tokenizer, size:int=16) -> None:
        '''
        Args:
            filepath: filepath to dataloader file
            tokenizer: tokenizer object
            size: how much context is taken into computation at a time
        '''
        with open(filepath, "r") as f:
            text = f.read()

        tokenizer.fit(text)
        self.tokens = tokenizer.encode(text)
        self.context_size = size

        windows = np.lib.stride_tricks.sliding_window_view(self.tokens, self.context_size + 1)

        self.contexts = windows[:,:-1]
        self.targets = windows[:,1:]

    def get_pairs(self, batch_size:int=32):
        """ 
        slice token per size as inputs
        """
        for i in range(0, len(self.targets), batch_size):
            yield(self.contexts[i:i+batch_size], self.targets[i:i+batch_size])