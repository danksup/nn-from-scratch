import json
from engine.model import Model
from engine.tokenizer import Tokenizer
from engine.embedding import Embedding
from engine.dataloader import DataLoader
import numpy as np

DEFAULT_CONFIGS = {
            "epochs": 100,
            "lr": 1e-3,
            "context_size": 64,
            "batch_size": 32,
            "seed": 42,
            "embed_dim":8,
            "base_width": 512,
            "dataset": "data/The_Expedition_of_Humphry_Clinker.txt"
        }

class Session:
    def __init__(self, model:Model,  tokenizer:Tokenizer, embedding:Embedding, configs:dict | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.embedding = embedding
        if configs is None:
            configs = {}

        self.configs = DEFAULT_CONFIGS | configs
        
    
    #TODO a mess
    @classmethod
    def build(cls, datapath:str):
        '''
        create a session with default model, tokenizer,embedding, and configuration.
        '''
        raise NotImplementedError("not yet")
        tokenizer = Tokenizer()
        tokenizer_len = len(tokenizer.chartoid)
        embedding = Embedding(tokenizer_len, 8)
        model = Model.build(DEFAULT_CONFIGS["context_size"] * DEFAULT_CONFIGS["embed_dim"],tokenizer_len , 2, DEFAULT_CONFIGS["base_width"])
        session = cls(model,tokenizer,embedding)
        session.embedding.embed_dim = session.configs["embed_dim"]

    @classmethod
    def build_from_files(cls):
        '''
        build from save file using filepath.
        '''
        raise NotImplementedError("not yet")

    def train(self,patience:int=10, display_message:bool=True):
        """
        Args:
            patience: max bad epochs
            display_message: obvious
        train using set configs
        """
        dataloader = DataLoader(self.configs["dataset"], self.tokenizer, self.configs["context_size"])
        if display_message:
            print(f"[TRAINING] param: {self.model.count_params()}, seed: {self.configs["seed"]}, epochs: {self.configs["epochs"]}, LR: {self.configs["lr"]}, embed dimension: {self.configs["embed_dim"]},  base_width: = {self.configs["base_width"]}, context_size: {self.configs["context_size"]}, batch_size: {self.configs["batch_size"]}")
        
        epoch = 0
        try:
            prev_error = None
            bad_epoch = 0
            for i in range(self.configs["epochs"]):
                epoch = i
                error = self.model.train(dataloader, self.embedding, lr=self.configs["lr"], batch_size=self.configs["batch_size"])

                if prev_error is not None:
                    too_small = abs(error - prev_error) < 0.001 
                    getting_worse = prev_error - error < 0

                    if too_small or getting_worse:
                        bad_epoch += 1
                        if display_message:
                            bad_epoch_reason = "improvement too small" if too_small else "getting worse"
                            print(f"epoch {i}: bad epoch: {bad_epoch_reason} | delta: {error-prev_error:.5f} | ({bad_epoch}/{patience})")
                    else:
                        if bad_epoch > 0 and display_message:
                            print(f"epoch: {i}: bad epoch count reset")
                        bad_epoch = 0

                    if bad_epoch >= patience:
                        if display_message:
                            print(f"training stopped at {i}, model isnt getting better")
                        break
                if i == self.configs["epochs"] - 1:
                    print(f"epoch {epoch} | avg loss: {error}")
                    
                prev_error = error
        except ValueError as e:
            print(f"epoch {epoch}: {e}")
            self.save("valueerror_save")
            raise
        except OverflowError as e:
            print(f"epoch {epoch}: {e}")
            self.save("overflow_save")
            raise
    
    def predict(self, context:np.ndarray):
        """
        predict next token
        """
        return self.model.predict(context, self.embedding)

    def save(self, filename:str, save_artifacts:bool=False):
        '''
        Args:
            filename: the name the save file will have. session_{filename}.json
            save_artifacts: also save artifacts seperately
        '''

        model_dict = self.model.to_dict()
        tokenizer_dict = self.tokenizer.to_dict()
        embedding_dict = self.embedding.to_dict()

        session = {
            "configs":self.configs,
            "model":model_dict,
            "tokenizer":tokenizer_dict,
            "embedding":embedding_dict,
        }

        filename = f"session_{filename}.json"
        with open(f"artifacts/sessions/{filename}", "w") as f:
            json.dump(session,f,indent=4)
    
    @classmethod
    def load(cls, filepath) -> "Session":
        """
        load the saved session file and build
        """
        with open(filepath, "r") as f:
            session = json.load(f)

        model = Model.from_dict(session["model"])
        embedding = Embedding.from_dict(session["embedding"])
        tokenizer = Tokenizer.from_dict(session["tokenizer"])
        configs = session["configs"]

        session = cls(model,tokenizer,embedding, configs=configs)
        return session