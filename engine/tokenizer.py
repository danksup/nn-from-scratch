import json
import numpy as np
class Tokenizer:
    def __init__(self):
        self.idtochar = {0:"<PAD>", 1:"<UNK>"}
        self.chartoid = {"<PAD>":0,"<UNK>":1}

    #TODO maybe what if i want a different method of splitting like maybe world level 
    def split_input(self, char:str) -> list[str]:
        """character level tokenizer"""
        return list(char)

    def fit(self, x:str) -> None:
        """
        Args:
            x: input

        fit x if not fitted.
        """

        tokens = self.split_input(x)
        for i in tokens:
            if i not in self.chartoid:
                next_id = len(self.chartoid)
                self.chartoid[i] = next_id
                self.idtochar[next_id] = i
        
    def encode(self, char:str) -> np.ndarray:
        """
        returns list of token id 
        """

        tokens = self.split_input(char)
        encoded = np.zeros(len(tokens),dtype=np.int64)
        for i,token in enumerate(tokens):
            if token not in self.chartoid:
                encoded[i] = (self.chartoid["<UNK>"])
            else:
                encoded[i] = (self.chartoid[token ])
        
        return encoded


    def decode(self, thing:list[int]) -> str:
        '''
        decode back list of id into char
        '''
        decoded = ""
        for i in thing:
            decoded += self.idtochar[i]

        return decoded
    
    def to_dict(self) -> dict:
        vocab = {
            "idtochar":self.idtochar,
            "chartoid":self.chartoid
        }
        return vocab

    def save(self, filename:str):
        """
        save the token
        """
        vocab = self.to_dict()
        filename = f"artifacts/tokens/tokenizer_char_level_{filename}.json"
        with open(filename, "w") as f:
            json.dump(vocab, f, indent=4)
    
    @classmethod
    def from_dict(cls,thing) -> "Tokenizer":
        tokenizer = cls()
        tokenizer.idtochar = {}
        tokenizer.chartoid = {}
        for token_id, char in thing["idtochar"].items():   
            tokenizer.idtochar[int(token_id)] = char
        for char, id in thing["chartoid"].items():   
            tokenizer.chartoid[char] = int(id)

        return tokenizer

    def load(self, filename:str) -> "Tokenizer":
        """
        load the token
        """
        try:
            with open(filename, 'r') as f:
                vocab = json.load(f)
            
            tokenizer = Tokenizer.from_dict(vocab)
            self.chartoid = tokenizer.chartoid
            self.idtochar = tokenizer.idtochar

            return self
        except FileNotFoundError:
            raise FileNotFoundError("file not found")
        except json.JSONDecodeError:
            raise ValueError("decode eror")
