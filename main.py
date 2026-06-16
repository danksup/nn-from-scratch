import random
SEED = 42
random.seed(SEED)
EPOCHS = 5000
LR = 1e-3

from engine.model import Model
from engine.layer import Layer

data = [(1, 1), (2, 4), (3, 9), (4, 16)]

print(data)
print(SEED)
print(EPOCHS)
print(LR)
model = Model()
model.add_layer(Layer(200, 1))
model.add_layer(Layer(2, 200))
model.add_layer(Layer(1, 2))
model.train(data, epochs=EPOCHS, lr=LR, print_loss=True)
print("params = ", model.count_params())