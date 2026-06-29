# nn-from-scratch
just a fun little project i do on my days off. a small llm project built for (my) learning purposes.

logs:
- older stuff (did not track)
- save/load (did not track)
- layernorm (did not track) (replaced by rmsnorm)
- attention (did not track)
- transformer (did not track)
- apple silicon acceleration (MLX) (jun 23 2026)
- rmsnorm (jun 24 2026)
- multi head attention (jun 24 2026)
- top p (jun 24 2026)
- weight tying (jun 24 2026)
- dropout (jun 25 2026)
- rope (jun 25 2026)
- train/valid split (jun 25 2026)
- mixed precision(jun 25 2026)
- swiglu(jun 26 2026)
- purely optimizing in june 27 2026
- purely optimizing in june 28 2026
- purely optimizing in june 29 2026
  
## Ongoing:
- fusing some operations
- optimizing optimizer
- adding comments, descriptions, typehints
    
## TODO (not in order):
- checkpoint
- lr scheduling
- better tokenization
- kv caching
- frequency penalty
- more optimization
  
#### Maybe:
- other gpus acceleration (maybe not)
- better batching


# Performance Logs
Apple M1 Pro \
param: 72512 | epochs: 1 | context_size: 64 | batch_size: 256 | embed_dim: 64 \
ff_width: 256 | optimizer: adamw | train_split: 0.9 | n_heads: 8 \
optimizer_args: {'lr': 0.001, 'beta': 0.9, 'beta2': 0.999, 'epsilon': 1e-08, 'weight_decay': 0.01} \
dataset: 5 files | using: MLX | block_size: 1 | corpus char len: 3417355 

- Date: 2026-06-28 | 1092171 function calls in 239.626 seconds | ram peaked at ~ 800MB | -
- Date: 2026-06-29 | 865328 function calls in 214.715 seconds | ram peaked at ~ 900MB | compiled all backwards and forwards into one each