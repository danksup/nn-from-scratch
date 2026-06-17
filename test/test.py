from engine.tokenizer import Tokenizer

t = Tokenizer()

t.encode("qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM[]\\;',./{}|:\"<>?~`1234567890!@#$%^&*()-=_+ ")
t.save("test")

u = Tokenizer().load("tokens/char_level_vocab_test.json")

print(u.chartoid)
print(u.idtochar)

x = u.encode("hello world")
print(x)
print(u.decode(x))