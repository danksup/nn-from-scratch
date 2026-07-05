import engine.backend as nx
from engine.sessions import Session

session = Session.load("artifacts/sessions/session_val_test_8256817151568.ram2n")
tokenizer = session.tokenizer
context_size = session.configs["context_size"]
context = "Question: "
print(f"input: {context}")
context = tokenizer.encode(context)
context = context.reshape(-1, context.shape[0])

TEMPERATURE = .8
TOP_K = 20
TOP_P = .8
N = 500
print(f"n: {N} | temp: {TEMPERATURE} | top_k: {TOP_K} | top_p: {TOP_P}")
session.inference(context, TEMPERATURE, TOP_K, TOP_P, N)