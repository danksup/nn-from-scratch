import os
backend = os.environ["BACKEND"] = "auto"
import random
# import mlx.core as mx
LR = 1e-3
EMBED_DIM = 128
CONTEXT_SIZE = 64
BATCH_SIZE = 48
BASE_WIDTH = 768 #4 * EMBED_DIM 
N_HEADS = EMBED_DIM // 8
N_KV_HEADS = N_HEADS// 2
VAL = .9

from pathlib import Path
import time

import cProfile
import pstats

from engine.transformer import Transformer
from engine.transformer_block import TransformerBlock
from engine.tokenizer import Tokenizer
from engine.embedding import Embedding
from engine.dataloader import DataLoader
from engine.sessions import Session
import engine.backend as nx

configs = {
            "context_size": CONTEXT_SIZE,
            "batch_size": BATCH_SIZE,
            "embed_dim":EMBED_DIM,
            "ff_width": BASE_WIDTH,
            "train_split":VAL,
            "n_heads": N_HEADS,
            "n_kv_heads": N_KV_HEADS,
            "optimizer":"adamw",
            "dataset":0,
            "optimizer_args":{
                "lr":LR,
                "beta":0.9,
                "beta2":0.999,
                "weight_decay":1e-2
            },
            "using":backend
        }

corpus = ""
tokenizer1 = Tokenizer.load("artifacts/tokenizer/tokenizer8192_33414037len.tokenizer")
# print(tokenizer1.vocab)
files = []
folder = Path("data")
for file in folder.iterdir():
    if file.name != ".gitkeep" and file.name[-1:-5:-1] == "txt." :
        files.append(file)

for file in files:
    with open(file) as f:
        data = f.read()
        corpus += data + "\n\n\n"

configs["dataset"] = f"{len(files)} files"
weight_n = CONTEXT_SIZE * EMBED_DIM
real_vocab_size = len(tokenizer1.vocab)

embedding1 = Embedding(real_vocab_size, EMBED_DIM)
tblock = TransformerBlock(EMBED_DIM, BASE_WIDTH,N_HEADS, N_KV_HEADS)
tblock2 = TransformerBlock(EMBED_DIM, BASE_WIDTH,N_HEADS, N_KV_HEADS)
tblock3 = TransformerBlock(EMBED_DIM, BASE_WIDTH,N_HEADS, N_KV_HEADS)
tblock4 = TransformerBlock(EMBED_DIM, BASE_WIDTH,N_HEADS, N_KV_HEADS)
transformer = Transformer(real_vocab_size,EMBED_DIM, "adamw")
transformer.add_block(tblock)
transformer.add_block(tblock2)
transformer.add_block(tblock3)
transformer.add_block(tblock4)
start = time.perf_counter()
configs["block_size"] = len(transformer.blocks)

print("loading dataloader", end="\r")
dataloader = DataLoader(corpus, tokenizer1, configs["context_size"])

corpus_len = len(corpus)
token_size = dataloader.tokens.size
ratio = ((corpus_len - token_size )/corpus_len) * 100
configs["corpus char len"] = f"{corpus_len} -> BPE compression ({len(tokenizer1.vocab)} vocab size): {token_size}. ratio = {ratio:.3f}% "
session1 = Session(transformer,tokenizer1,embedding1, configs)

# profiler = cProfile.Profile()
# profiler.enable()
start = time.perf_counter()
# mx.metal.start_capture("transformer.gputrace")
session1.benchmark(dataloader, 100)
end = time.perf_counter()
# mx.metal.stop_capture()
print(f"benchmarking finished. time: {end - start:.3f}s")

# profiler.disable()
# stats = pstats.Stats(profiler)
# stats.sort_stats("cumtime")
# stats.print_stats(100)




