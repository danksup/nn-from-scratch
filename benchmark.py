import os
backend = os.environ["BACKEND"] = "auto"
import random
# import mlx.core as mx
EMBED_DIM = 128
CONTEXT_SIZE = 256
BATCH_SIZE = 64
BASE_WIDTH = 1024#4 * EMBED_DIM 
N_HEADS = EMBED_DIM // 8
N_KV_HEADS = N_HEADS // 2
WINDOWS = CONTEXT_SIZE // 8
N_EXPERTS = 24
CF = 1.25
VAL = .9
TOP_K = 2

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
            "MoE":{
                "topk":2,
                "cf":CF,
                "n_experts":N_EXPERTS,
                "ff_width":BASE_WIDTH
            },
            "optimizer":"adamw",
            "train_split":.9,
            "n_heads": N_HEADS,
            "n_kv_heads":N_KV_HEADS,
            "windows":WINDOWS,
            "train_split":VAL,
            "optimizer_args":{
                "min_lr":1e-4,
                "max_lr":1e-2,
                "beta":0.9,
                "beta2":0.999,
                "epsilon":1e-8,
                "weight_decay":0.01
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
tblock = TransformerBlock(EMBED_DIM ,BASE_WIDTH,N_HEADS, N_KV_HEADS, N_EXPERTS, CF, W=WINDOWS)
tblock2 = TransformerBlock(EMBED_DIM, BASE_WIDTH,N_HEADS, N_KV_HEADS, N_EXPERTS, CF, W=WINDOWS)
tblock3 = TransformerBlock(EMBED_DIM, BASE_WIDTH,N_HEADS, N_KV_HEADS, N_EXPERTS, CF, W=WINDOWS)
tblock4 = TransformerBlock(EMBED_DIM, BASE_WIDTH,N_HEADS, N_KV_HEADS, N_EXPERTS, CF, W=WINDOWS)
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
session1.benchmark(dataloader, 5)
end = time.perf_counter()
# mx.metal.stop_capture()
print(f"benchmarking finished. time: {end - start:.3f}s")

# profiler.disable()
# stats = pstats.Stats(profiler)
# stats.sort_stats("cumtime")
# stats.print_stats(100)




