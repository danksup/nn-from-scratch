import engine.backend as nx
from engine.sessions import Session

session = Session.load("artifacts/sessions/session_val_test_7924562451116.ram2n")
tokenizer = session.tokenizer
context_size = session.configs["context_size"]
context = "Question: who does the voice of carl in phineas and ferb?"
print(f"input: {context}")
context = tokenizer.encode(context)
context = context.reshape(-1, context.shape[0])

TEMPERATURE = .9
TOP_K = 40
TOP_P = .9
N = 50
print(f"n: {N} | temp: {TEMPERATURE} | top_k: {TOP_K} | top_p: {TOP_P}")
session.inference(context, TEMPERATURE, TOP_K, TOP_P, N)
