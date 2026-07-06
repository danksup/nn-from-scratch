import engine.backend as nx
from engine.sessions import Session

session = Session.load("artifacts/sessions/session_val_test_8994356491990.ram2n")
tokenizer = session.tokenizer
context_size = session.configs["context_size"]
context = "Question: A man made lake is known as what?"
print(f"input: {context}")
context = tokenizer.encode(context)
context = context.reshape(-1, context.shape[0])

TEMPERATURE = .7
TOP_K = 20
TOP_P = .8
N = 500
print(f"n: {N} | temp: {TEMPERATURE} | top_k: {TOP_K} | top_p: {TOP_P}")
session.inference(context, TEMPERATURE, TOP_K, TOP_P, N)
