from engine.tokenizer import Tokenizer
from engine.embedding import Embedding
from engine.dataloader import DataLoader

u = Tokenizer().load("tokens/char_level_vocab_test.json")
farewell = "/Users/rama/Desktop/project1/data/a_farewell_to_arms.txt"
print(u.decode([95]))
a = DataLoader(farewell, u)

