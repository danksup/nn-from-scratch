from engine.tokenizer import Tokenizer

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

        self.tokens = tokenizer.encode(text)
        self.context_size = size

    def get_pairs(self, batch_size:int=32):
        """ 
        slice token per size as inputs
        """
        batch = []
        for i in range(self.context_size, len(self.tokens)):
            context = self.tokens[i- self.context_size : i]
            next_token = self.tokens[i]
            batch.append([context, next_token])
            if len(batch) == batch_size:
                yield batch
                batch = []
        
        if batch:
            yield batch