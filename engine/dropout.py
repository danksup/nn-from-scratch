from engine.backend import nx

class Dropout:
    def __init__(self,p) -> None:
        self.is_training:bool=False
        self.p = p
        pass