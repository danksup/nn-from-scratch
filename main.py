import random
SEED = 42
random.seed(SEED)
EPOCHS = 1
LR = 0.005
EMBED_DIM = 8
CONTEXT_SIZE = 32
BATCH_SIZE = 32
BASE_WIDTH = EMBED_DIM * CONTEXT_SIZE

import time


from engine.model import Model
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
            "base_width": BASE_WIDTH,
            "dataset": "data/The_Expedition_of_Humphry_Clinker.txt",
            "optimizer_args":{
                "lr":0.01,
                "beta":0.5
            }
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
session1.train(display_message=True)
end = time.time()
print(f"training finished. time: {end - start:.3f}s")
session1.save("test_momentum")

# session_load = Session.load("/Users/rama/Desktop/project1/artifacts/sessions/session_test_momentum.json")
# context = session_load.tokenizer.encode("The nature of religion is questi")

# for _ in range(400):
    
#     predicted_id = session_load.predict(context, top_k=10, temperature=0.5)
#     print(session_load.tokenizer.decode([predicted_id]), end="")
#     context = context[1:] + [predicted_id]  
