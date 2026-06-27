import json
from engine.backend import nx
import re
from typing import Any

class Tokenizer:
    def __init__(self):
        self.idtochar = {0:"<PAD>", 1:"<UNK>", 2: "<EOT>"}
        self.chartoid = {"<PAD>":0,"<UNK>":1, "<EOT>":2}

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
        
    def encode(self, char: str):
        tokens = self.split_input(char)
        encoded = [self.chartoid.get(token, self.chartoid["<UNK>"])for token in tokens]
        return nx.array(encoded, dtype=nx.int32)
        # return encoded


    def decode(self, thing: Any) -> str:
        decoded = ""

        for token_id in thing:
            decoded += self.idtochar[int(token_id)]

        return decoded
    
    def to_dict(self) -> dict:
        vocab = {
            "idtochar":self.idtochar,
            "chartoid":self.chartoid
        }
        return vocab
    
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
