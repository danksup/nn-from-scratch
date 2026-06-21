import random
SEED = 42
random.seed(SEED)
EPOCHS = 1
LR = 1e-3
EMBED_DIM = 8
CONTEXT_SIZE = 32
BATCH_SIZE = 32
# BASE_WIDTH = (EMBED_DIM * CONTEXT_SIZE ) // 4
BASE_WIDTH = 16

#not added yet to session
PATIENCE = 20
TRESHOLD = 1e-2

import time
import numpy as np


from engine.transformer import Transformer
from engine.tokenizer import Tokenizer
from engine.embedding import Embedding
from engine.dataloader import DataLoader
from engine.sessions import Session

configs = {
            "epochs": EPOCHS,
            "context_size": CONTEXT_SIZE,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "embed_dim":EMBED_DIM,
            "ff_width": BASE_WIDTH,
            "optimizer":"adamw",
            "dataset": "data/The_Expedition_of_Humphry_Clinker.txt",
            "optimizer_args":{
                "lr":LR,
                "beta":0.9,
                "beta2":0.999,
                "weight_decay":1e-2
            }
        }

data = "/Users/rama/Desktop/project1/data/The_Expedition_of_Humphry_Clinker.txt"
tokenizer1 = Tokenizer()
dataloader = DataLoader(data, tokenizer1, CONTEXT_SIZE)
vocab_size = len(tokenizer1.chartoid)
print(vocab_size)
weight_n = CONTEXT_SIZE * EMBED_DIM
embedding1 = Embedding(vocab_size, EMBED_DIM)
# model1 = Transformer.build(weight_n,vocab_size,2, BASE_WIDTH)
# print(model1)
transformer = Transformer(vocab_size,EMBED_DIM, "adamw")
session1 = Session(transformer,tokenizer1,embedding1, configs)
start = time.time()
session1.train(display_message=True)
end = time.time()
print(f"training finished. time: {end - start:.3f}s")
# session1.save("test_pe")

# session_load = Session.load("/Users/rama/Desktop/project1/artifacts/sessions/session_test_adamw_1e-3.json")
# context = session_load.tokenizer.encode("The nature of religion is questi")
# print(len(context))
# TEMPERATURE = 0.3
# TOP_K = 10
# print(f"temperature={TEMPERATURE}")
# print(f"top_k={TOP_K}")
# for _ in range(400):
#     context_batch = context.reshape(1, -1)
#     predicted_id = session_load.predict(context_batch, top_k=TOP_K, temperature=TEMPERATURE)
#     print(session_load.tokenizer.decode([predicted_id]), end="", flush=True)
#     context = np.append(context[1:], predicted_id)
# print()