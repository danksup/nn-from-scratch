from misc.misc_embedding import n_closest
from engine.tokenizer import Tokenizer
from engine.sessions import Session

session = Session.load("artifacts/sessions/session_val_test_8705484269019.ram2n")
tokenizer = session.tokenizer
embedding = session.embedding

closest_to = "mythology"
print(f"closest to {closest_to}:")
nclose = n_closest(closest_to, tokenizer, embedding)
