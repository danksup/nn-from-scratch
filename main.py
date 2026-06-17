import random
SEED = 42
random.seed(SEED)
EPOCHS = 1
LR = 1e-3
EMBED_DIM = 4
HIDDEN_N = 64


from engine.model import Model
from engine.layer import Layer
from engine.tokenizer import Tokenizer
from engine.embedding import Embedding
from engine.dataloader import DataLoader


# u = Tokenizer().load("tokens/char_level_vocab_test.json")
# farewell = "/Users/rama/Desktop/project1/data/a_farewell_to_arms.txt"
# a = DataLoader(farewell, u)
# vocab_size = len(u.chartoid)
# embedding = Embedding(vocab_size, EMBED_DIM)
# weight_n = a.context_size * EMBED_DIM
# layer1 = Layer(HIDDEN_N, weight_n)
# layer2 = Layer(vocab_size, HIDDEN_N, activation=None, activation_derivative=None)
# model = Model()
# model.add_layer(layer1)
# model.add_layer(layer2)

# trained = model.train(a,embedding,EPOCHS,print_loss=True)
# model.save("farewell_limited_to_10kchars")
# embedding.save("chocolate")

saved_model = Model().load("models/model_10855_farewell_limited_to_10kchars.json")
u = Tokenizer().load("tokens/char_level_vocab_test.json")
embedding = Embedding(1,1).load("embeds/chocolate.json")
context = u.encode("can i taste your")  # 16 chars
predicted_id = saved_model.predict(context, embedding)
print(u.decode([predicted_id]))