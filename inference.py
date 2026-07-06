import engine.backend as nx
from engine.sessions import Session

session = Session.load("artifacts/sessions/session_val_test_5843783971268.ram2n")
tokenizer = session.tokenizer
context_size = session.configs["context_size"]
context = "what was the occupation of gandhi if it werent for us to be enslaved?"
print(f"input: {context}")
context = tokenizer.encode(context)
context = context.reshape(-1, context.shape[0])

TEMPERATURE = 0.7
TOP_K = 10
TOP_P = .9
N = 500
print(f"n: {N} | temp: {TEMPERATURE} | top_k: {TOP_K} | top_p: {TOP_P}")
session.inference(context, TEMPERATURE, TOP_K, TOP_P, N)
