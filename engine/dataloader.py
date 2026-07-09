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
        indices = nx.permutation(len(self.contexts))
        shuffled_contexts = self.contexts[indices] #type:ignore
        shuffled_targets = self.targets[indices]#type:ignore
        split = int(len(self.contexts) * self.train_split)
        self.train_contexts =shuffled_contexts[:split]
        self.train_targets = shuffled_targets[:split]
        self.validate_contexts = shuffled_contexts[split:]
        self.validate_targets = shuffled_targets[split:]

    def get_pairs(self, batch_size:int=32):
        """ 
        slice token per size as inputs
        """
        train_indices = nx.permutation(len(self.train_contexts))
        shuffled_contexts = self.train_contexts[train_indices]
        shuffled_targets = self.train_targets[train_indices]
        for i in range(0, len(shuffled_contexts), batch_size):
            yield(shuffled_contexts[i:i+batch_size], shuffled_targets[i:i+batch_size])
    
    def get_validation_pairs(self, batch_size:int=32):
        """ 
        slice token per size as inputs
        """
        for i in range(0, len(self.validate_targets), batch_size):
            yield(self.validate_contexts[i:i+batch_size], self.validate_targets[i:i+batch_size])