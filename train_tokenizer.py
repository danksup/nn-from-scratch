from pathlib import Path
import time
from engine.tokenizer import Tokenizer
import cProfile
import pstats
VOCAB_SIZE = 4096

corpus = ""
files = []
folder = Path("data")
for file in folder.iterdir():
    if file.name != ".gitkeep" and file.name[-1:-5:-1] == "txt." :
        files.append(file)

for file in files:
    with open(file) as f:
        data = f.read()
        corpus += data + "\n\n\n"

print(len(corpus))
tokenizer1 = Tokenizer(VOCAB_SIZE)
t = int(time.time())
print()
start = time.perf_counter()
# profiler = cProfile.Profile()
# profiler.enable()
a =tokenizer1.fit(corpus)
# profiler.disable()
# stats = pstats.Stats(profiler)
# stats.sort_stats("cumtime")
# stats.print_stats(100)
end = time.perf_counter()
tokenizer_save_name = f"{VOCAB_SIZE}_{len(corpus)}len"
tokenizer1.save(tokenizer_save_name)
print(f"{tokenizer_save_name} saved. fitting finished in {end-start:.3f}")