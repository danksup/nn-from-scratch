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
    def get_pairs(word:list[str]|tuple[str,...]) -> list[tuple[str,...]]:
        pairs = []
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            pairs.append(pair)
        return pairs
    
    @staticmethod
    def get_pair_counts( word_counts:dict) -> Counter:
        counts = Counter()
        for word, count in word_counts.items():
            for pair in Tokenizer.get_pairs(word):
                counts[pair] += count
        return counts

    @staticmethod
    def build_pair_index(word_counts:dict) -> tuple[Counter, dict]:
        counts = Counter()
        pair_to_words = {}
        for word, count in word_counts.items():
            for pair in Tokenizer.get_pairs(word):
                counts[pair] += count
                if pair not in pair_to_words:
                    pair_to_words[pair] = set()
                pair_to_words[pair].add(word)
        
        return counts, pair_to_words

    @staticmethod
    def remove_word(word:tuple[str,...], freq:int, pair_counts:Counter[tuple[str,...]], pair_to_words:dict[tuple[str,...],set[tuple[str,...]]]):
        word_pairs = Tokenizer.get_pairs(word)
        for pair in word_pairs:
            pair_counts[pair] -= freq

        set_pairs = set(word_pairs)
        for pair in set_pairs:
            if pair_counts[pair] <= 0:
                del pair_counts[pair]

        for pair in set_pairs:
            pair_to_words[pair].remove(word)

            if not pair_to_words[pair]:
                pair_to_words.pop(pair)
    
    @staticmethod
    def add_word(word:tuple[str,...], freq:int, pair_counts:Counter[tuple[str,...]], pair_to_words:dict[tuple[str,...],set[tuple[str,...]]]):
        word_pairs = Tokenizer.get_pairs(word)
        
        for pair in word_pairs:
            pair_counts[pair] += freq

        for pair in set(word_pairs):
            if pair not in pair_to_words:
                pair_to_words[pair] = set()

            pair_to_words[pair].add(word)

    @staticmethod
    def merge(word:tuple, best_pair):
        i = 0
        n = len(word)
        found = False

        while i < n - 1:
            pair = word[i], word[i + 1]
            if pair == best_pair:
                found = True
                break
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
            return new_word
        else:
            return word


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
        pair_counts, pair_to_words = self.build_pair_index(word_counts)

        while len(self.vocab) < self.target_vocab_size:     
            if not pair_counts:
                break
            best_pair = pair_counts.most_common(1)[0][0]
            affected_words = pair_to_words[best_pair].copy()

            for affected_word in affected_words:
                freq = word_counts.pop(affected_word)
                self.remove_word(affected_word, freq, pair_counts, pair_to_words)
                merged = tuple(self.merge(affected_word, best_pair))
                word_counts[merged] = word_counts.get(merged, 0) + freq
                self.add_word(merged, freq, pair_counts, pair_to_words)

            merged_best = best_pair[0] + best_pair[1]

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