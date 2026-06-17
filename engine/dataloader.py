from engine.tokenizer import Tokenizer

class DataLoader:
    def __init__(self,filepath:str, tokenizer:Tokenizer, size:int=16) -> None:
        with open(filepath, "r") as f:
            text = f.read()

        self.tokens = tokenizer.encode(text[:10000])
        self.context_size = size

    def get_pairs(self):
        """ 
        slice token per size as inputs
        """
        for i in range(self.context_size, len(self.tokens)):
            context = self.tokens[i- self.context_size : i]
            next_token = self.tokens[i]
            yield(context, next_token)