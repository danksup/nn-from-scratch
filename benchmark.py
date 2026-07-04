import os
backend = os.environ["BACKEND"] = "mlx"
import random
import mlx.core as mx
EPOCHS = 1
LR = 1e-3
EMBED_DIM = 32
CONTEXT_SIZE = 32
BATCH_SIZE = 256
BASE_WIDTH = 4 * EMBED_DIM 
N_HEADS = EMBED_DIM // 8
N_KV_HEADS = N_HEADS//2
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
            "epochs": EPOCHS,
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
tokenizer1 = Tokenizer.load("artifacts/tokenizer/session_4096_1783114959.tokenizer")
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
transformer = Transformer(real_vocab_size,EMBED_DIM, "adamw")
transformer.add_block(tblock)
start = time.perf_counter()
configs["block_size"] = len(transformer.blocks)
dataloader = DataLoader(corpus, tokenizer1, configs["context_size"])
configs["corpus char len"] = dataloader.tokens.size
session1 = Session(transformer,tokenizer1,embedding1, configs)

profiler = cProfile.Profile()
profiler.enable()
# start = time.perf_counter()
# mx.metal.start_capture("transformer.gputrace")
session1.benchmark(dataloader, 50)
end = time.perf_counter()
# mx.metal.stop_capture()
# print(f"training finished. time: {end - start:.3f}s")

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats("cumtime")
stats.print_stats(100)




