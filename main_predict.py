from engine.backend import nx
from engine.sessions import Session
session_load = Session.load("/Users/rama/Desktop/project1/artifacts/sessions/session_2_blocks_3023129550237.ram2n")
context = session_load.tokenizer.encode("Sherlock Holmes Watson")
print(f"context: {session_load.tokenizer.decode(context.tolist())} | {len(context)}")
TEMPERATURE = .7
TOP_K = 40
TOP_P = .9
print(f"temperature={TEMPERATURE}")
print(f"top_k={TOP_K}")
print(f"top_p={TOP_P}")
for _ in range(200):
    context_batch = context.reshape(1, -1)
    predicted_id = session_load.predict(context_batch, top_k=TOP_K, temperature=TEMPERATURE, top_p=TOP_P)
    print(session_load.tokenizer.decode([predicted_id]), end="", flush=True)
    new_token = nx.array([predicted_id]).astype(nx.int64)
    context = nx.concatenate([context[1:], new_token])
print()

#dropout test
# context_batch = context.reshape(1, -1)
# a = predicted_id = session_load.predict(context_batch, top_k=TOP_K, temperature=TEMPERATURE, top_p=TOP_P)
# b = predicted_id = session_load.predict(context_batch, top_k=TOP_K, temperature=TEMPERATURE, top_p=TOP_P)
# c = predicted_id = session_load.predict(context_batch, top_k=TOP_K, temperature=TEMPERATURE, top_p=TOP_P)
# print(a,b,c)
# # print(session_load.tokenizer.decode([predicted_id]), end="", flush=True)
# # new_token = nx.array([predicted_id]).astype(nx.int64)
# # context = nx.concatenate([context[1:], new_token])

