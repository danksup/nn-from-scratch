import engine.backend as nx
from collections import Counter
import heapq as heap
from typing import Any
import pickle


class Tokenizer:
    def __init__(self, target_vocab_size= 1024):
        self.target_vocab_size = target_vocab_size
        self.merge_rank = {}
        self.id_to_token = {0:"<PAD>", 1:"<UNK>", 2: "<EOT>", 3:"</w>"}
        self.vocab = {"<PAD>":0,"<UNK>":1, "<EOT>":2, "</w>":3}

    def init_vocab(self, corpus:str) -> None:
        for char in corpus:
            if char.isspace():
                continue
            if char not in self.vocab:
                next_id = len(self.vocab)
                self.vocab[char] = next_id
                self.id_to_token[next_id] = char

    def word_to_ids(self, word: str):
        return [self.vocab[ch] for ch in word] + [self.vocab["</w>"]]

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
    def get_pairs(word):
        for i in range(len(word) - 1):
            yield (word[i], word[i + 1])
    
    @staticmethod
    def get_pair_counts( word_counts:dict) -> Counter:
        counts = Counter()
        for word, count in word_counts.items():
            for pair in Tokenizer.get_pairs(word):
                counts[pair] += count
        return counts
    
    @staticmethod
    def get_best_pair(pair_heap, pair_counts):
        while pair_heap:
            neg_count, pair = pair_heap[0]
            current_count = pair_counts.get(pair, 0)
            if current_count == -neg_count:
                return pair
            heap.heappop(pair_heap)

        return None

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
    def remove_word(word:tuple[str,...], freq:int, pair_counts:Counter[tuple[str,...]], pair_to_words:dict[tuple[str,...],set[tuple[str,...]]], pair_heap):
        word_pairs = list( Tokenizer.get_pairs(word))
        set_pairs = set(word_pairs)

        for pair in word_pairs:
            pair_counts[pair] -= freq
            
        for pair in set_pairs:
            if pair_counts[pair] > 0:
                heap.heappush(pair_heap, (-pair_counts[pair], pair))
            else:
                if pair in pair_counts:
                    del pair_counts[pair]

        for pair in set_pairs:
            pair_to_words[pair].remove(word)
            if not pair_to_words[pair]:
                pair_to_words.pop(pair)
    
    @staticmethod
    def add_word(word:tuple[str,...], freq:int, pair_counts:Counter[tuple[str,...]], pair_to_words:dict[tuple[str,...],set[tuple[str,...]]], pair_heap):
        word_pairs = list( Tokenizer.get_pairs(word))
        
        for pair in word_pairs:
            pair_counts[pair] += freq 
           
        for pair in set(word_pairs):
            if pair_counts[pair] > 0:
                heap.heappush(pair_heap, (-pair_counts[pair], pair))
                if pair not in pair_to_words:
                    pair_to_words[pair] = set()

                pair_to_words[pair].add(word)

    @staticmethod
    def merge(word:list, best_pair, new_id):
        i = 0
        n = len(word)
        new_word = []
        while i < n - 1:
            pair = word[i], word[i + 1]
            if pair == best_pair:
                new_word.append(new_id)
                i += 2
            else:
                new_word.append(word[i])
                i+=1
        if i == n - 1:
            new_word.append(word[n-1])

        return new_word

    def fit(self, corpus:str):
        self.init_vocab(corpus)
        words = [self.word_to_ids(word) for word in corpus.split()]


        word_counts = self.get_word_counts(words)
        pair_counts, pair_to_words = self.build_pair_index(word_counts)

        pair_heap = [(-count, pair) for pair, count in pair_counts.items()]
        heap.heapify(pair_heap)

        while len(self.vocab) < self.target_vocab_size:     
            if not pair_counts:
                break
            best_pair = self.get_best_pair(pair_heap, pair_counts)
            if best_pair is None:
                break

            affected_words = pair_to_words[best_pair].copy()

            new_id = len(self.vocab)
            for affected_word in affected_words:
                freq = word_counts.pop(affected_word)
                self.remove_word(affected_word, freq, pair_counts, pair_to_words, pair_heap)
                merged = tuple(self.merge(affected_word, best_pair, new_id))
                word_counts[merged] = word_counts.get(merged, 0) + freq
                self.add_word(merged, freq, pair_counts, pair_to_words,pair_heap)

            merged_best = self.id_to_token[best_pair[0]] + self.id_to_token[best_pair[1]]

            len_vocab = len(self.vocab)
            self.merge_rank[best_pair] = (len(self.merge_rank), new_id)
            self.vocab[merged_best] = len_vocab
            self.id_to_token[len_vocab] = merged_best
        
    def encode(self, text: str) -> nx.ArrayLike:
        words = [self.word_to_ids(word) for word in text.split()]

        for idx, word in enumerate(words):
            while True:
                best_pair = None
                best_rank = float("inf")
                best_new_id = None

                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])

                    if pair in self.merge_rank:
                        rank, new_id = self.merge_rank[pair]

                        if rank < best_rank:
                            best_rank = rank
                            best_pair = pair
                            best_new_id = new_id

                if best_pair is None:
                    break

                word = self.merge(word, best_pair, best_new_id)
                words[idx] = word

        tokens = [token for word in words for token in word]
        encoded = tokens
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
        filename = f"tokenizer{filename}.tokenizer"
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