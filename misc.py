from misc.misc_embedding import n_closest, embedding_of
from engine.sessions import Session

session = Session.load("artifacts/sessions/session_val_test_8994356491990.ram2n")
tokenizer = session.tokenizer
embedding = session.embedding

a = embedding_of("woman",tokenizer, embedding)
b = embedding_of("husband", tokenizer, embedding)
d = a + b 
n_closest(d, tokenizer, embedding)

