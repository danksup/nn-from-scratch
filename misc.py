from misc.misc_embedding import n_closest
from engine.sessions import Session

session = Session.load("artifacts/sessions/session_val_test_8256817151568.ram2n")
tokenizer = session.tokenizer
embedding = session.embedding

closest_to = "destroy"
print(f"closest to {closest_to}:")
nclose = n_closest(closest_to, tokenizer, embedding)

