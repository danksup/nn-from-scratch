import os
os.environ["USE_BACKEND"] = "auto"
import random

SEED = 42
EPOCHS = 10
LR = 1e-3
EMBED_DIM = 64
CONTEXT_SIZE = 64
BATCH_SIZE = 256
BASE_WIDTH = 4 * EMBED_DIM 
N_HEADS = 8
VAL = .9

#not hooked yet to session
PATIENCE = 20
TRESHOLD = 1e-2

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
from engine.sessions import nx

configs = {
            "epochs": EPOCHS,
            "context_size": CONTEXT_SIZE,
            "batch_size": BATCH_SIZE,
            "embed_dim":EMBED_DIM,
            "ff_width": BASE_WIDTH,
            "train_split":VAL,
            "n_heads": N_HEADS,
            "optimizer":"adamw",
            "dataset":0,
            "optimizer_args":{
                "lr":LR,
                "beta":0.9,
                "beta2":0.999,
                "weight_decay":1e-2
            },
            "using":nx.backend
        }

corpus = ""
tokenizer1 = Tokenizer()
files = []
folder = Path("data")
for file in folder.iterdir():
    if file.name != ".gitkeep":
        files.append(file)

for file in files:
    with open(file) as f:
        data = f.read()
        corpus += data + "\n\n\n"

tokenizer1.fit(corpus)

configs["dataset"] = f"{len(files)} files"
vocab_size = len(tokenizer1.chartoid)
weight_n = CONTEXT_SIZE * EMBED_DIM
embedding1 = Embedding(vocab_size, EMBED_DIM)
tblock = TransformerBlock(EMBED_DIM, BASE_WIDTH,N_HEADS)
tblock2 = TransformerBlock(EMBED_DIM, BASE_WIDTH,N_HEADS)
transformer = Transformer(vocab_size,EMBED_DIM, "adamw")
transformer.add_block(tblock)
transformer.add_block(tblock2)
configs["block_size"] = len(transformer.blocks)
session1 = Session(transformer,tokenizer1,embedding1, configs)
dataloader = DataLoader(corpus, tokenizer1, configs["context_size"])

a = random.randint(1,9999999999999)
a = str(a)
profiler = cProfile.Profile()
profiler.enable()
# start = time.time()
session1.train(dataloader, display_message=True)
# end = time.time()
# print(f"training finished. time: {end - start:.3f}s")

profiler.disable()
session1.save(f"val_test_{a}")

stats = pstats.Stats(profiler)
stats.sort_stats("cumtime")
stats.print_stats(100)


