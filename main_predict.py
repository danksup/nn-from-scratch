from engine.backend import nx
from engine.sessions import Session

# session_load = Session.load("artifacts/sessions/4blocks.ram2n")
# context = session_load.tokenizer.encode("Hello!")
# print(f"context: {session_load.tokenizer.decode(context.tolist())} | {len(context)}")
# TEMPERATURE = .8
# TOP_K = 30
# TOP_P = .9
# print(f"temperature={TEMPERATURE}")
# print(f"top_k={TOP_K}")
# print(f"top_p={TOP_P}")
# for _ in range(300):
#     context_batch = context.reshape(1, -1)
#     predicted_id = session_load.predict(context_batch, top_k=TOP_K, temperature=TEMPERATURE, top_p=TOP_P)
#     print(session_load.tokenizer.decode([predicted_id]), end="", flush=True)
#     new_token = nx.array([predicted_id]).astype(nx.int64)
#     context = nx.concatenate([context[1:], new_token])
# print()


