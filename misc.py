from misc.misc_embedding import n_closest#, embedding_of
from engine.sessions import Session

session = Session.load("artifacts/sessions/session_val_test_8256817151568.ram2n")
tokenizer = session.tokenizer
embedding = session.embedding


