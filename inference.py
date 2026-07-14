import engine.backend as mx
from engine.sessions import Session

# session = Session.load("artifacts/sessions/session_38974464_params_20_epochs.ram2ns")
# session.configs["epochs"] = 20
# tokenizer = session.tokenizer
# context_size = session.configs["context_size"]
# context = "what the fucks"
# print(f"input: {context}")
# context = tokenizer.encode(context)
# context = context.reshape(-1, context.shape[0])

# TEMPERATURE = 0.6
# TOP_K = 20
# TOP_P = .8
# N = 200
# print(f"n: {N} | temp: {TEMPERATURE} | top_k: {TOP_K} | top_p: {TOP_P}")
# session.inference(context, TEMPERATURE, TOP_K, TOP_P, N)

a = mx.array([[29,12,3,1],[29,12,30,1],[29,120,3,1],[21,12,3,1]], dtype=mx.int32)
b = mx.argsort(a)
c = mx.topk(a,2)
d = mx.topk(a, 2, return_element=True)
print(b[..., ::-1][...,:-2])
print(c)
print(d)