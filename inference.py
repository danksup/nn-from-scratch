import engine.backend as nx
from engine.sessions import Session

session = Session.load("artifacts/sessions/2block.ram2n")
tokenizer = session.tokenizer
context_size = session.configs["context_size"]
print(context_size)
context = "Hello!"
print(f"context: {context}")
context = tokenizer.encode(context)
context = context.reshape(-1, context.shape[0])

TEMPERATURE = 0.8
TOP_K = 20
TOP_P = .9
N = 400
print(f"n: {N} | temp: {TEMPERATURE} | top_k: {TOP_K} | top_p: {TOP_P}")
infered = session.inference(context, TEMPERATURE, TOP_K, TOP_P, N)
infered = infered.reshape(infered.shape[0])
decoded = tokenizer.decode(infered)
print(decoded, flush=True)