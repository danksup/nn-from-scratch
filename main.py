import random
SEED = 42
random.seed(SEED)
EPOCHS = 10
LR = 1e-3
EMBED_DIM = 4
width = 16
CONTEXT_SIZE = 16
BATCH_SIZE = 32

import time


from engine.model import Model
from engine.layer import Layer
from engine.tokenizer import Tokenizer
from engine.embedding import Embedding
from engine.dataloader import DataLoader

u = Tokenizer()
data = "/Users/rama/Desktop/project1/data/The_Expedition_of_Humphry_Clinker.txt"
a = DataLoader(data, u, CONTEXT_SIZE)
vocab_size = len(u.chartoid)
embedding = Embedding(vocab_size, EMBED_DIM)
weight_n = a.context_size * EMBED_DIM
layer1 = Layer(width//2, weight_n)
layer2 = Layer(width//4, width//2)
layer3 = Layer(vocab_size, width//4, activation=None, activation_derivative=None)
model = Model()
model.add_layer(layer1)
model.add_layer(layer2)
model.add_layer(layer3)
print(f"param: {model.count_params()}, seed: {SEED}, epochs: {EPOCHS}, LR: {LR}, embed dimension: {EMBED_DIM}, width: = {width}, context_size: {CONTEXT_SIZE}, batch_size: {BATCH_SIZE}")

start = time.time()
trained = model.train(a,embedding,EPOCHS,print_loss=True, batch_size=BATCH_SIZE)
end = time.time()

model.save(f"humpy_limited_to_full_{EPOCHS}epochs")
embedding.save("chocolate")
u.save("more")
print(f"training time: {end - start:.3f}s")


# saved_model = Model().load("models/model_76371_humpy_limited_to_full_5epochs.json")
# u = Tokenizer().load("tokens/char_level_vocab_more.json")
# embedding = Embedding(1,1).load("embeds/chocolate.json")
# context = u.encode("The old lighthouse watched storms roll across the dark bay. End ")
# print(f"context: {u.decode(context)}")
# # full_prob = []
# for _ in range(20):
#     full, predicted_id = saved_model.predict(context, embedding)
#     print(u.decode([predicted_id]), end="")
#     context = context[1:] + [predicted_id]  
#     # full_prob.append(full)
# # print("\n")
# # for i in range(len(full_prob)):
# #     a = []
# #     for j in range(len(full_prob[i])):
# #         a.append((u.idtochar[j], f"{full_prob[i][j]:.3f}"))
    
# #     print(a)
    

