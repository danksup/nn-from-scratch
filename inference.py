import engine.backend as mx
from engine.sessions import Session

# session = Session.load("artifacts/sessions/session_2393088_params.ram2n")
# tokenizer = session.tokenizer
# context_size = session.configs["context_size"]
# context = "jajajajajajajajaja"
# print(f"input: {context}")
# context = tokenizer.encode(context)
# context = context.reshape(-1, context.shape[0])

# TEMPERATURE = 0.6
# TOP_K = 20
# TOP_P = .8
# N = 200
# print(f"n: {N} | temp: {TEMPERATURE} | top_k: {TOP_K} | top_p: {TOP_P}")
# session.inference(context, TEMPERATURE, TOP_K, TOP_P, N)
import numpy as np
# import mlx.core as mx

E = 3
B = 1
T = 4
N = B*T
a = mx.array([1,1,2,0], mx.int32)

routing_mask = mx.zeros((N, E))
# routing_mask = mx.where(a, routing_mask, 1)
row = mx.arange(N, dtype=mx.int32)
# routing_mask = routing_mask.at[row,a].add(1)
routing_mask  = mx.add_at(routing_mask,(row,a), 1)
print(routing_mask)
b = mx.cumsum(routing_mask, axis=0)
print(b)