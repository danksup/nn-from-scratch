from engine.tokenizer import Tokenizer
from engine.embedding import Embedding

t = Tokenizer()

t.encode("qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM[]\\;',./{}|:\"<>?~`1234567890!@#$%^&*()-=_+ ")
t.save("test")

u = Tokenizer().load("tokens/char_level_vocab_test.json")



x = u.encode("hello world")
print(len(x))
a = Embedding(len(u.chartoid), 8).forward(x)