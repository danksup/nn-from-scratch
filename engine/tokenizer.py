import json
import engine.backend as nx
import re
from collections import Counter
from typing import Any

class Tokenizer:
    def __init__(self, vocab_size= 1024):
        self.vocab_size = vocab_size
        self.merges = {}
        self.id_to_token = {0:"<PAD>", 1:"<UNK>", 2: "<EOT>", 3:"</w>"}
        self.vocab = {"<PAD>":0,"<UNK>":1, "<EOT>":2, "<w/>":3}

    @staticmethod
    def get_pair_counts( words:list) -> Counter:
        counts = Counter()
        for word in words:
           for i in range(len(word) - 1):
                pair = (word[i], word[i+1])
                counts[pair] += 1
        return counts

    @staticmethod
    def merge_pair(words:list, best_pair:str):
        merged = []
        for word in words:
            i = 0
            n = len(word)
            new_word = []
            while i < n - 1:
                pair = word[i], word[i + 1]
                if pair == best_pair:
                    new_word.append(word[i] + word[i+1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i+=1
            if i == n - 1:
                new_word.append(word[n-1])
            merged.append(new_word)
        return merged

    def fit(self, corpus:str):
        words = [list(word) +  ["</w>"]  for word in corpus.split()]
        
        for word in words:
            for i in word:
                if i not in self.vocab:
                    next_id = len(self.vocab)
                    self.vocab[i] = next_id
                    self.id_to_token[next_id] = i

        while len(self.vocab) < self.vocab_size:
            counts = self.get_pair_counts(words)
            if not counts:
                break
            
            best_pair = counts.most_common(1)[0][0]
            merged_best = best_pair[0] + best_pair[1]

            words = self.merge_pair(words, best_pair)
            
            self.merges[best_pair] = merged_best
            self.vocab[merged_best] = len(self.vocab)
            self.id_to_token[len(self.vocab)-1] = merged_best
        
    def encode(self, text: str):
        words = [list(word) +  ["</w>"]  for word in text.split()]

        for pair, merged in self.merges.items():
            words = self.merge_pair(words, pair)

        tokens = [token for word in words for token in word]
        encoded = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]
        return nx.array(encoded, dtype=nx.int32)

    def decode(self, thing: Any) -> str:
        decoded = ""

        for token_id in thing:
            if int(token_id) == self.vocab["<PAD>"]:
                continue
            decoded += self.id_to_token[int(token_id)]

        decoded = decoded.replace("</w>", " ")
        return decoded
    
    def to_dict(self) -> dict:
        vocab = {
            "merges":self.merges,
            "vocab":self.vocab,
            "id_to_token":self.id_to_token
        }
        return vocab
    
    @classmethod
    def from_dict(cls,thing) -> "Tokenizer":
        tokenizer = cls()

        tokenizer.vocab = thing["vocab"]
        tokenizer.id_to_token = thing["id_to_token"]
        tokenizer.merges = thing["merges"]

        return tokenizer
