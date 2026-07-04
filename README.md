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
- KV caching (jun 30 2026)
- grouped query attention (jun 30 2026)
- BPE tokenizer (jul 1 2026)
- rewrote some BPE functions in C (jul 2 2026)
- use hashing for counting on C BPE (jul 2 2026)
- revert back to pythonic BPE (jul 2 2026)
- optimizing BPE in jul 3 2026
- incremental BPE in jul 4 2026
- logsumexp crossentropy in jul 4 2026
  
## Ongoing:
- optimizing
  
## TODO (not in order):
- checkpoint
- lr scheduling
- frequency penalty
- more optimization
  
#### Maybe:
- other gpus acceleration (maybe not)
- better batching

## Bugs:
### Fixed
- inference degradation after a certain number of tokens (fixed jul 1 2026) -> fix: sliding kv position
### Open
- inference breaks when token length is too large -> generate freqs on demand when needed (jun 29 2026)
  
# Performance Logs
Apple M1 Pro \
param: 72512 | epochs: 1 | context_size: 64 | batch_size: 256 | embed_dim: 64 \
ff_width: 256 | optimizer: adamw | train_split: 0.9 | n_heads: 8 \
optimizer_args: {'lr': 0.001, 'beta': 0.9, 'beta2': 0.999, 'epsilon': 1e-08, 'weight_decay': 0.01} \
dataset: 5 files | using: MLX | block_size: 1 | corpus char len: 3417355 

- Date: 2026-06-28 | 1092171 function calls in 239.626 seconds | ram peaked at ~ 800MB | compiled each layer backward an forward locally
- Date: 2026-06-29 | 865328 function calls in 214.715 seconds | ram peaked at ~ 900MB | block level compilation
- Date: 2026-06-29 | 504964 function calls in 211.726 seconds | ram peaked at ~ 800MB | compiled optimizers
#### slightly different configs, the rest is the same unless otherwise stated
- Date 2026-06-30 | 623763 function calls in 204.493 seconds | ram peaked at ~ 800MB | grouped-query attention (8Q/4KV)
- Date 2026-07-04 | 131387 function calls in 153.710 seconds | ram peaked at ~ 800MB | logsumexp cross entropy, compiled cross entropy; BPE Tokenizer, corpus char len: 1106747 param: 323712 (larger vocabulary because of BPE)

### Tokenizer
- 11296590 corpus len | 2048 | fitting finished in 1614.249 py 
- 11296590 corpus len | 2048 | fitting finished in 579.541 C
- 11296590 corpus len | 2048 | fitting finished in 629.932 post optimized py (3 jul 2026)
- 11296590 corpus len | 2048 | fitting finished in 17.050 incremental BPE py (4 jul 2026)