from pathlib import Path

SEED = 42
EPOCHS = 1
LR = 1e-3
EMBED_DIM = 8
CONTEXT_SIZE = 32
BATCH_SIZE = 256
BASE_WIDTH = 4 * EMBED_DIM 
BASE_WIDTH = 16


#not added yet to session
PATIENCE = 20
TRESHOLD = 1e-2

import time
import cProfile
import pstats


from engine.transformer import Transformer
from engine.transformer_block import TransformerBlock
from engine.tokenizer import Tokenizer
from engine.embedding import Embedding
from engine.dataloader import DataLoader
from engine.sessions import Session

configs = {
            "epochs": EPOCHS,
            "context_size": CONTEXT_SIZE,
            "batch_size": BATCH_SIZE,
            "embed_dim":EMBED_DIM,
            "ff_width": BASE_WIDTH,
            "optimizer":"adamw",
            "dataset":0,
            "optimizer_args":{
                "lr":LR,
                "beta":0.9,
                "beta2":0.999,
                "weight_decay":1e-2
            }
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
tblock = TransformerBlock(EMBED_DIM, BASE_WIDTH)
transformer = Transformer(vocab_size,EMBED_DIM, "adamw")
transformer.add_block(tblock)
session1 = Session(transformer,tokenizer1,embedding1, configs)
dataloader = DataLoader(corpus, tokenizer1, configs["context_size"])



profiler = cProfile.Profile()
profiler.enable()
# start = time.time()
session1.train(dataloader, display_message=True)
# end = time.time()
profiler.disable()
# print(f"training finished. time: {end - start:.3f}s")
session1.save("test_")

stats = pstats.Stats(profiler)
stats.sort_stats("cumtime")
stats.print_stats(40)


