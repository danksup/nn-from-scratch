import ctypes
from pathlib import Path

_LIB_PATH = Path(__file__).parent / "build" / "libtokenizer.dylib"
lib = ctypes.CDLL(_LIB_PATH)
class VocabEntry(ctypes.Structure):
    _fields_ = [
        ("token", ctypes.c_char_p),
        ("id", ctypes.c_int),
    ]

class Vocab(ctypes.Structure):
    _fields_ = [
        ("entries", ctypes.POINTER(VocabEntry)),
        ("count", ctypes.c_int),
        ("capacity", ctypes.c_int),
    ]

class MergeRule(ctypes.Structure):
    _fields_ = [
        ("first", ctypes.c_char_p),
        ("second", ctypes.c_char_p),
        ("merged", ctypes.c_char_p),
        ("rank", ctypes.c_int),
    ]

class MergeList(ctypes.Structure):
    _fields_ = [
        ("rules", ctypes.POINTER(MergeRule)),
        ("count", ctypes.c_int),
        ("capacity", ctypes.c_int),
    ]

class Tokenizer(ctypes.Structure):
    _fields_ = [
        ("vocab", Vocab),
        ("mergelist", MergeList),
    ] 

lib.fit_from_words.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.c_int, ctypes.c_int]
lib.fit_from_words.restype = Tokenizer

def fit_native(corpus:str, target_vocab_size):    
    vocab = {}
    id_to_token = {}
    merge_rank = {}

    corpus = corpus.encode("ascii", "ignore").decode("ascii")
    words = corpus.split()
    words = [w.encode("utf-8") for w in words]
    arr = (ctypes.c_char_p * len(words))(*words)
    tokenizer = lib.fit_from_words(arr, len(words), target_vocab_size)
    for i in range(tokenizer.vocab.count):
        entry = tokenizer.vocab.entries[i]
        token = entry.token.decode("utf-8")
        token_id = entry.id

        vocab[token] = token_id
        id_to_token[token_id] = token


    for i in range(tokenizer.mergelist.count):
        rule = tokenizer.mergelist.rules[i]

        first = rule.first.decode("utf-8")
        second = rule.second.decode("utf-8")
        merged = rule.merged.decode("utf-8")
        rank = rule.rank

        merge_rank[(first, second)] = (rank, merged)
    
    return vocab, id_to_token, merge_rank

