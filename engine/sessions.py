from engine.transformer import Transformer
from engine.tokenizer import Tokenizer
from engine.embedding import Embedding
from engine.dataloader import DataLoader
from engine.activations import softmax
from engine.optimizer import AdamW
from engine.backend import nx
import time
import pickle

DEFAULT_CONFIGS = {
            "epochs": 100,
            "context_size": 64,
            "batch_size": 32,
            "embed_dim":8,
            "ff_width": 512,
            "optimizer":"adamw",
            "optimizer_args":{
                "lr":0.05,
                "beta":0.9,
                "beta2":0.999,
                "epsilon":1e-8,
                "weight_decay":0.01
            }
        }

OPTIMIZERS = {
    # "sgd": SGD,
    # "momentum": Momentum,
    "adamw": AdamW,
    "adam": AdamW,
}

class Session:
    def __init__(self, transformer:Transformer,  tokenizer:Tokenizer, embedding:Embedding, configs:dict | None = None,init_optimizer:bool=True):
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.embedding = embedding
        if configs is None:
            configs = {}

        self.configs = DEFAULT_CONFIGS | configs
        self.configs["optimizer_args"] = DEFAULT_CONFIGS["optimizer_args"] | configs.get("optimizer_args", {})

        if init_optimizer: #if the optimizers need to be optimized first, otherwise from_dict params would get overridden
            optimizer_class = OPTIMIZERS[self.configs["optimizer"]]
            transformer.optimizer = optimizer_class(**self.configs["optimizer_args"])

        

    @classmethod
    def build_from_files(cls):
        '''
        build from save file using filepath.
        '''
        raise NotImplementedError("not yet")
    
    def count_params(self):
        """
        whole architecture number of params
        """
        total = 0
        for i in self.transformer.blocks:
            total += i.ff1.weights.size
            total += i.ff2.weights.size
            total += i.ff1.biases.size
            total += i.ff2.biases.size
            total += i.attention.Wq.size
            total += i.attention.Wk.size
            total += i.attention.Wv.size
            total += i.attention.Wo.size
            total += i.rmsnorm1.gamma.size
            total += i.rmsnorm2.gamma.size
        total += self.embedding.lookup_table.size
        total += self.transformer.classifier.weights.size
        total += self.transformer.classifier.biases.size
        return total

    def train(self,dataloader:DataLoader,patience:int=10, display_message:bool=True):
        """
        Args:
            patience: max bad epochs
            display_message: obvious
        train using set configs
        """
        if display_message:
            t_mess = f"[TRAINING]param: {self.count_params()} "
            for key,val in self.configs.items():
                t_mess += f"{key}: {val} "
            print(t_mess)        
        epoch = 0
        try:
            prev_error = None
            bad_epoch = 0
            # checkpoint = None
            # best_loss = None
            THRESHOLD = 1e-4
            for i in range(self.configs["epochs"]):
                epoch = i
                start = time.perf_counter()
                error = self.transformer.train(dataloader, self.embedding, batch_size=self.configs["batch_size"])
                nx.eval(error)
                end = time.perf_counter()

                time_ = end-start

                if prev_error is not None:
                    improvement = prev_error - error
                    too_small = improvement > 0 and improvement < THRESHOLD
                    getting_worse = improvement < 0

                    if too_small or getting_worse:
                        bad_epoch += 1
                        if display_message:
                            bad_epoch_reason = "improvement too small" if too_small else "getting worse"
                            print(f"epoch {i}: bad epoch: {bad_epoch_reason} | delta: {error-prev_error:.5f} | ({bad_epoch}/{patience}) | time: {time_}")
                    else:
                        #not how this works, checkpoint just references self so...
                        # checkpoint = self
                        # best_loss = error
                        if bad_epoch > 0:
                            if display_message:
                                print(f"epoch: {i}: bad epoch count reset")
                        bad_epoch = 0

                    if bad_epoch >= patience:
                        if display_message:
                            print(f"[STOPPED] epoch {i} | not getting better | avg loss: {error}")
                            # self.save("save_stopped")
                        break
                display_every = max(1, self.configs["epochs"] // 10)
                if display_message and( i % display_every == 0 or i == self.configs["epochs"] - 1):
                    
                    print(f"epoch {epoch} | avg loss: {error} | time: {time_}")
                    # largest = 0

                    # for layer in self.model.layers:
                    #     largest = max(
                    #         largest,
                    #         np.max(np.abs(layer.weights))
                    #     )

                    # print("max weight:", largest)
                    
                prev_error = error
        except ValueError as e:
            print(f"epoch {epoch}: {e}")
            # self.save("valueerror_save")
            raise
        except OverflowError as e:
            print(f"epoch {epoch}: {e}")
            # self.save("overflow_save")
            raise
    
    def predict(self, context, temperature=0.8, top_k=3):
        logits = self.transformer.predict(context, self.embedding)

        probs = softmax(logits / temperature)[0, -1]

        top_k = min(top_k, len(probs))

        top_indices = nx.argpartition(probs, -top_k)[-top_k:]

        top_probs = probs[top_indices]
        top_probs /= nx.sum(top_probs)

        return nx.random_choice(top_indices,p=top_probs)

    def save(self, filename:str, save_artifacts:bool=False):
        '''
        Args:
            filename: the name the save file will have. session_{filename}.json
            save_artifacts: also save artifacts seperately
        '''

        transformer_dict = self.transformer.to_dict()
        tokenizer_dict = self.tokenizer.to_dict()
        embedding_dict = self.embedding.to_dict()

        session = {
            "configs":self.configs,
            "transformer":transformer_dict,
            "tokenizer":tokenizer_dict,
            "embedding":embedding_dict,
            "optimizer_states":self.transformer.optimizer.to_dict()
        }

        filename = f"session_{filename}.ram2n"
        with open(f"artifacts/sessions/{filename}", "wb") as f:
           f.write(b"RAM2N")
           f.write((1).to_bytes(4, "little"))
           pickle.dump(session, f)
    
    @classmethod
    def load(cls, filepath) -> "Session":
        """
        load the saved session file and build
        """
        with open(filepath, "rb") as f:
            magic = f.read(5)
            if magic != b"RAM2N":
                raise ValueError("unknown file")
            version = int.from_bytes(f.read(4), "little")
            session = pickle.load(f)

        transformer = Transformer.from_dict(session["transformer"])
        embedding = Embedding.from_dict(session["embedding"])
        tokenizer = Tokenizer.from_dict(session["tokenizer"])
        configs = session["configs"]
        optimizer_class = OPTIMIZERS[configs["optimizer"]]
        transformer.optimizer = optimizer_class.from_dict(session["optimizer_states"])
        
        return  cls(transformer,tokenizer,embedding, configs=configs, init_optimizer=False)