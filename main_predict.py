import numpy as np
from engine.sessions import Session

session_load = Session.load("/Users/rama/Desktop/project1/artifacts/sessions/session_test_.ram2n")
context = session_load.tokenizer.encode("The nature of religion is questionable")
print(f"context: {session_load.tokenizer.decode(context.tolist())} | {len(context)}")
TEMPERATURE = .7
TOP_K = 5
print(f"temperature={TEMPERATURE}")
print(f"top_k={TOP_K}")
for _ in range(100):
    context_batch = context.reshape(1, -1)
    predicted_id = session_load.predict(context_batch, top_k=TOP_K, temperature=TEMPERATURE)
    print(session_load.tokenizer.decode([predicted_id]), end="", flush=True)
    context = np.append(context[1:], predicted_id)
print()