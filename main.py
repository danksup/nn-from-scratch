import random
SEED = 42
random.seed(SEED)
EPOCHS = 100
LR = 1e-3
EMBED_DIM = 4
BASE_WIDTH = 16
CONTEXT_SIZE = 16
BATCH_SIZE = 32

import time


from engine.model import Model
from engine.layer import Layer
from engine.tokenizer import Tokenizer
from engine.embedding import Embedding
from engine.dataloader import DataLoader
from engine.sessions import Session

configs = {
            "epochs": EPOCHS,
            "lr": LR,
            "context_size": CONTEXT_SIZE,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "embed_dim":EMBED_DIM,
            "base_width": BASE_WIDTH,
            "dataset": "data/The_Expedition_of_Humphry_Clinker.txt"
        }
data = "/Users/rama/Desktop/project1/data/The_Expedition_of_Humphry_Clinker.txt"
tokenizer1 = Tokenizer()
dataloader = DataLoader(data, tokenizer1, CONTEXT_SIZE)
vocab_size = len(tokenizer1.chartoid)
weight_n = CONTEXT_SIZE * EMBED_DIM
embedding1 = Embedding(vocab_size, EMBED_DIM)
model1 = Model.build(weight_n,vocab_size,2, BASE_WIDTH)
print(model1)
session1 = Session(model1,tokenizer1,embedding1, configs)
start = time.time()
session1.train()
end = time.time()
print(f"training finished. time: {end - start:.3f}s")
session1.save("test")

context = tokenizer1.encode("The old lighthouse watched storms roll across the dark bay. End ")
# for _ in range(20):
#     full, predicted_id = session1.predict(context)
#     print(tokenizer1.decode([predicted_id]), end="")
#     context = context[1:] + [predicted_id]  
