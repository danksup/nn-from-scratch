from engine.tokenizer import Tokenizer

tokenizer = Tokenizer.load("artifacts/tokenizer/tokenizer16384_195605563len.tokenizer")
print(tokenizer.vocab)