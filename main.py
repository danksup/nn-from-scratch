import random
SEED = 42
random.seed(SEED)
EPOCHS = 10
LR = 1e-3
EMBED_DIM = 32
CONTEXT_SIZE = 64
BATCH_SIZE = 16
# BASE_WIDTH = (EMBED_DIM * CONTEXT_SIZE ) // 4
BASE_WIDTH = 128
DATA = "data/a_farewell_to_arms.txt"


#not added yet to session
PATIENCE = 20
TRESHOLD = 1e-2

import time
import numpy as np


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
            "seed": SEED,
            "embed_dim":EMBED_DIM,
            "ff_width": BASE_WIDTH,
            "optimizer":"adamw",
            "dataset": DATA,
            "optimizer_args":{
                "lr":LR,
                "beta":0.9,
                "beta2":0.999,
                "weight_decay":1e-2
            }
        }

tokenizer1 = Tokenizer()
dataloader = DataLoader(DATA, tokenizer1, CONTEXT_SIZE)
vocab_size = len(tokenizer1.chartoid)
weight_n = CONTEXT_SIZE * EMBED_DIM
embedding1 = Embedding(vocab_size, EMBED_DIM)
tblock = TransformerBlock(EMBED_DIM, BASE_WIDTH)
transformer = Transformer(vocab_size,EMBED_DIM, "adamw")
transformer.add_block(tblock)
session1 = Session(transformer,tokenizer1,embedding1, configs)
start = time.time()
session1.train(display_message=True)
end = time.time()
print(f"training finished. time: {end - start:.3f}s")
session1.save("test_")

# session_load = Session.load("/Users/rama/Desktop/project1/artifacts/sessions/session_test_.ram2n")
# context = session_load.tokenizer.encode("we must")
# print(f"context: {session_load.tokenizer.decode(context.tolist())} | {len(context)}")
# TEMPERATURE = 1
# TOP_K = 5
# print(f"temperature={TEMPERATURE}")
# print(f"top_k={TOP_K}")
# for _ in range(100):
#     context_batch = context.reshape(1, -1)
#     predicted_id = session_load.predict(context_batch, top_k=TOP_K, temperature=TEMPERATURE)
#     print(session_load.tokenizer.decode([predicted_id]), end="", flush=True)
#     context = np.append(context[1:], predicted_id)
# print()
