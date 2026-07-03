import engine.backend as nx
from collections import Counter
from typing import Any
import pickle


class Tokenizer:
    def __init__(self, target_vocab_size= 1024):
        self.target_vocab_size = target_vocab_size
        self.merge_rank = {}
        self.id_to_token = {0:"<PAD>", 1:"<UNK>", 2: "<EOT>", 3:"</w>"}
        self.vocab = {"<PAD>":0,"<UNK>":1, "<EOT>":2, "</w>":3}

    @staticmethod 
    def get_word_counts(words:list) -> dict:
        counter = {}
        for word in words:
            tword = tuple(word)
            if tword in counter:
                counter[tword] += 1
            else:
                counter[tword] = 1
        
        return counter

    @staticmethod
    def get_pair_counts( word_counts:dict) -> Counter:
        counts = Counter()
        for word, count in word_counts.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i+1])
                counts[pair] += 1 * count
        return counts

    @staticmethod
    def merge(word:list, best_pair):
        i = 0
        n = len(word)
        found = False

        while i < n - 1:
            pair = word[i], word[i + 1]
            if pair == best_pair:
                found = True
                break
            found = False
            i += 1

        i = 0
        if found == True:
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
            return (new_word)
        else:
            return(word)


    @staticmethod
    def merge_pair(words:list, best_pair:tuple):
        merged = []
        for word in words:
            merged.append(Tokenizer.merge(word, best_pair))
        return merged

    def fit(self, corpus:str):
        words = [list(word) +  ["</w>"]  for word in corpus.split()]
        for word in words:
            for i in word:
                if i not in self.vocab:
                    next_id = len(self.vocab)
                    self.vocab[i] = next_id
                    self.id_to_token[next_id] = i

        word_counts = self.get_word_counts(words)
        while len(self.vocab) < self.target_vocab_size:
            counts = self.get_pair_counts(word_counts)
            if not counts:
                break
            
            best_pair = counts.most_common(1)[0][0]
            merged_best = best_pair[0] + best_pair[1]
            
            new_word_count = {}
            for tword, freq in word_counts.items():
                lword = list(tword)
                merged = tuple(self.merge(lword, best_pair))
                new_word_count[merged] = new_word_count.get(merged, 0) + freq
            word_counts = new_word_count

            len_vocab = len(self.vocab)
            self.merge_rank[best_pair] = (len(self.merge_rank), merged_best)
            self.vocab[merged_best] = len_vocab
            self.id_to_token[len_vocab] = merged_best
        
    def encode(self, text: str):
        words = [list(word) +  ["</w>"]  for word in text.split()]

        for idx, word in enumerate(words):
            while True:
                best_pair = None
                best_rank = float("inf")

                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])

                    if pair in self.merge_rank:
                        rank, merged = self.merge_rank[pair]

                        if rank < best_rank:
                            best_rank = rank
                            best_pair = pair

                if best_pair is None:
                    break

                word = self.merge_pair([word], best_pair)[0]
                words[idx] = word

        tokens = [token for word in words for token in word]
        encoded = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]
        return nx.array(encoded, dtype=nx.int32)

    def decode(self, thing:Any) -> str:
        decoded = ""

        for token_id in thing:
            if int(token_id) == self.vocab["<PAD>"]:
                continue
            decoded += self.id_to_token[int(token_id)]

        decoded = decoded.replace("</w>", " ")
        return decoded
    
    def to_dict(self) -> dict:
        vocab = {
            "merge_rank":self.merge_rank,
            "vocab":self.vocab,
            "id_to_token":self.id_to_token
        }
        return vocab
    
    @classmethod
    def from_dict(cls,thing) -> "Tokenizer":
        tokenizer = cls()

        tokenizer.vocab = thing["vocab"]
        tokenizer.id_to_token = thing["id_to_token"]
        tokenizer.merge_rank = thing["merge_rank"]

        return tokenizer
    
    def save(self, filename):
        tokenizer = self.to_dict()
        filename = f"session_{filename}.tokenizer"
        with open(f"artifacts/tokenizer/{filename}", "wb") as f:
            f.write(b"tokenizer")
            f.write((1).to_bytes(4, "little"))
            pickle.dump(tokenizer, f)
    
    @classmethod
    def load(cls, filepath) -> "Tokenizer":
        with open(filepath, "rb") as f:
            magic = f.read(9)
            if magic != b"tokenizer":
                raise ValueError("unknown file")
            version = int.from_bytes(f.read(4), "little")
            loaded = pickle.load(f)

        
        tokenizer = cls.from_dict(loaded)

        return tokenizer