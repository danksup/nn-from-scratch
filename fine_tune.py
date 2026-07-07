from helper.singleton import init_corpus
from engine.sessions import Session
from engine.dataloader import DataLoader
PATH_TO_SESSION = "artifacts/sessions/session_checkpoint_save.ram2n"

corpus, _ = init_corpus("data")
print(corpus)
session = Session.load(PATH_TO_SESSION)
dataloader = DataLoader(corpus, session.tokenizer)
session.train(dataloader)