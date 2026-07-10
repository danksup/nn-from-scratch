import engine.backend as nx
from engine.sessions import Session

session = Session.load("artifacts/sessions/session_2393088_params.ram2n")
tokenizer = session.tokenizer
context_size = session.configs["context_size"]
context = "a singer in a smokey room. a smell of wine and cheap perfume."
print(f"input: {context}")
context = tokenizer.encode(context)
context = context.reshape(-1, context.shape[0])

TEMPERATURE = 0.6
TOP_K = 20
TOP_P = .8
N = 200
print(f"n: {N} | temp: {TEMPERATURE} | top_k: {TOP_K} | top_p: {TOP_P}")
session.inference(context, TEMPERATURE, TOP_K, TOP_P, N)
