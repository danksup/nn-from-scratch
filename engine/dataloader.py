from engine.tokenizer import Tokenizer
import engine.backend as nx

class DataLoader:
    def __init__(self,data:str, tokenizer:Tokenizer, context_size:int=16, train_split=0.9) -> None:
        '''
        Args:
            data: corpus
            tokenizer: tokenizer object
            context_size: how much context is taken into computation at a time
            train_split: split contexts between training and validation
        '''
        self.train_split = train_split
        self.tokens = tokenizer.encode(data)
        self.context_size = context_size

        windows = nx.sliding_window_view(self.tokens, self.context_size + 1)

        self.contexts = windows[:,:-1]
        self.targets = windows[:,1:]

        split = int(len(self.contexts) * self.train_split)
        self.train_contexts = self.contexts[:split]
        self.train_targets = self.targets[:split]
        self.validate_contexts = self.contexts[split:]
        self.validate_targets = self.targets[split:]


    def get_pairs(self, batch_size:int=32):
        """ 
        slice token per size as inputs
        """
        for i in range(0, len(self.train_targets), batch_size):
            yield(self.train_contexts[i:i+batch_size], self.train_targets[i:i+batch_size])
    
    def get_validation_pairs(self, batch_size:int=32):
        """ 
        slice token per size as inputs
        """
        for i in range(0, len(self.validate_targets), batch_size):
            yield(self.validate_contexts[i:i+batch_size], self.validate_targets[i:i+batch_size])