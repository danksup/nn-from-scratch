class Tensor:
    def __init__(self) -> None:
        self.parent = None
        self.creator = None
        self.gradient = 0.0

class Something:
    def multiply