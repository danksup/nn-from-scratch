from misc.misc_embedding import n_closest, embedding_of
from engine.sessions import Session

session = Session.load("artifacts/sessions/session_val_test_5324538083613.ram2n")
tokenizer = session.tokenizer
embedding = session.embedding

closest_to = "block"
print(f"closest to {closest_to}")
n_closest(closest_to, tokenizer, embedding)
