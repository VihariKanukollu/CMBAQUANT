## CMBA GPU v1 – Quick Recipes
=================================================================================

### TL;DR (in simple words)
- MCP is the switchboard. You set upper limits (ceilings). For every input, MCP decides how much compute to spend (steps, cycles, width, features). Easy inputs use less; hard inputs use more.
- You mostly turn on MCP and set ceilings. Everything else is automatic.

### Quick start (minimal flags)
```bash
!python pretrain.py \
  data_path=<DATA> epochs=8 eval_interval=1 global_batch_size=32 \
  lr=5e-5 lr_warmup_steps=200 lr_min_ratio=0.1 weight_decay=0.12 \
  arch=CMBA_GPU
```
- This already enables dynamic cycles/steps/width and skips heavy features at eval when not needed.

=================================================================================
### MCP central feature control (suduko)
=================================================================================


!TORCHDYNAMO_DISABLE=1 python3 pretrain.py \
  data_path=data/sudoku-6x6-1000 epochs=1000 eval_interval=40 \
  global_batch_size=32 lr=7e-5 lr_warmup_steps=40 lr_min_ratio=0.1 weight_decay=0.12 \
  arch=CMBA_GPU eval_halt_max_steps=96 arch.ponder_eval_best_step=true

=================================================================================
### MCP central feature control (chat)
=================================================================================

!TORCHDYNAMO_DISABLE=1 python3 pretrain.py \
  +data_mode=chat +tokenizer_id=mistralai/Mistral-7B-Instruct-v0.2 +trust_remote_code=true \
  +chat_train_jsonl=/workspace/CMBAQUANT/HRM/data/chat/train.jsonl \
  +chat_eval_jsonl=/workspace/CMBAQUANT/HRM/data/chat/val.jsonl \
  +chat_max_seq_len=1024 \
  arch=CMBA_GPU \
  global_batch_size=16 epochs=21 eval_interval=7 \
  eval_halt_max_steps=16 arch.halt_max_steps=16 arch.halt_min_steps_ceiling=2 \
  arch.ponder_eval_best_step=false checkpoint_every_eval=true

=================================================================================
### MCP central feature control (ARC2)
=================================================================================

Standard run (full eval, sane defaults):

TORCHDYNAMO_DISABLE=1 python3 pretrain.py \
  data_path=data/arc-2-aug-1000 \
  epochs=480 eval_interval=40 \
  global_batch_size=32 lr=7e-5 lr_warmup_steps=40 lr_min_ratio=0.1 weight_decay=0.12 \
  arch=CMBA_GPU \
  eval_halt_max_steps=96 arch.halt_max_steps=128 arch.halt_min_steps_ceiling=4 arch.halt_min_steps=1 \
  checkpoint_every_eval=true

Fast sanity run (quick checks):

TORCHDYNAMO_DISABLE=1 python3 pretrain.py \
  data_path=data/arc-2-aug-1000 \
  epochs=3 eval_interval=1 \
  global_batch_size=32 lr=7e-5 lr_warmup_steps=40 lr_min_ratio=0.1 weight_decay=0.12 \
  arch=CMBA_GPU \
  eval_max_batches=4 eval_halt_max_steps=48 arch.halt_max_steps=96 arch.halt_min_steps=1 \
  checkpoint_every_eval=true

=================================================================================
### MCP central feature control (recommended)
=================================================================================

- MCP owns all feature switches; you set ceilings, MCP allocates effort.
```bash
!python pretrain.py data_path=<DATA> epochs=8 eval_interval=1 \
  global_batch_size=32 lr=5e-5 lr_warmup_steps=200 lr_min_ratio=0.1 weight_decay=0.12 arch=hrm_quant_v1 \
  arch=CMBA_GPU \
  arch.mcp_backend=mlp arch.mcp_temp=0.7 arch.mcp_hard_eval=true \
  arch.mcp_auto_features=true arch.mcp_feature_profile=balanced arch.mcp_eval_threshold=0.5 \
  +arch.mcp_feature_keys=[puzzle,halt,gate,headbias,routing,film,rope,sched,ponder,ntm,h_cycles,l_cycles,mlp_expand,heads_active,min_steps,max_steps] \
  # Ceilings for dynamic compute
  +arch.max_h_cycles=3 +arch.max_l_cycles=3 +arch.max_expansion=4.0 +arch.halt_min_steps_ceiling=6 \
  # Optional per-feature costs (encourage cheaper paths)
  +arch.mcp_feature_costs.ntm=1e-4 +arch.mcp_feature_costs.routing=5e-5 +arch.mcp_feature_costs.film=5e-5
```
### High-level
- This runs training and lets MCP (our “switchboard”) decide how much computation to use per input, within ceilings you set. Easy inputs use less; hard inputs use more.

### Training basics
- data_path, epochs, eval_interval: where data is, how long to train, how often to evaluate.
- global_batch_size, lr, lr_warmup_steps, lr_min_ratio, weight_decay: standard optimizer/training knobs.

### Model
- arch=hrm_quant_v1; arch.name=hrm.cmba_gpu_v1_live@HierarchicalReasoningModel_ACTV1: use CMBA v1 implementation.

### MCP core
- arch.mcp_enabled=true: turn on MCP gates.
- arch.mcp_backend=mlp: gates produced by a tiny MLP.
- arch.mcp_temp=0.7: gate sampling temperature during training (lower → crisper gates).
- arch.mcp_hard_eval=true: at eval, gates are hard-thresholded (on/off).
- arch.mcp_auto_features=true: if a gate is “off,” skip that feature’s compute at eval.
- arch.mcp_feature_profile=balanced: initial bias for gates (fast | balanced | algorithmic).
- arch.mcp_eval_threshold=0.5: gate threshold at eval (≥ 0.5 = ON).

#### Profile presets: `fast` | `balanced` | `algorithmic`
- fast: Start biased toward cheaper compute.
  - Fewer cycles/steps on average, narrower MLP, heavy features (NTM/routing/film) biased OFF early.
  - Best when you want maximum throughput; accuracy may rise slower but MCP can still turn features ON if needed.
- balanced: Neutral starting point (default).
  - No strong bias; MCP learns where to spend compute from scratch.
- algorithmic: Start biased toward more compute.
  - More cycles/steps, wider MLP, and heavy features biased ON early (NTM/routing/scheduler/ponder).
  - Best for hard reasoning tasks where extra compute pays off.
Note: This only seeds initial biases; training will override if the data disagrees.

### What MCP controls (mcp_feature_keys)
- puzzle: blend between puzzle embedding sources.
- halt: mixes into halt-vs-continue Q-values.
- gate: scales attention/MLP residuals.
- headbias: per-head attention scaling.
- routing: token routing (keep fewer tokens).
- film: FiLM conditioning.
- rope: RoPE phase adjust.
- sched: H↔L scheduler gate.
- ponder: influences Ponder behavior (KL weight, eval halt threshold).
- ntm: external memory (read/write).
- h_cycles, l_cycles: number of H/L reasoning cycles (1..max).
- mlp_expand: MLP width factor (1..max).
- heads_active: reserved for head-group gating (future).
- min_steps, max_steps: min/max outer steps per sample (within ceilings).

### Ceilings for dynamic compute
- max_h_cycles=3, max_l_cycles=3: upper bounds for H/L cycles; MCP picks 1..3.
- max_expansion=4.0: upper bound on MLP width; MCP picks 1..4×.
- halt_min_steps_ceiling=6: upper bound on the minimum step floor; MCP picks 0..6.

### Cost shaping (optional)
- mcp_feature_costs.ntm=1e-4, .routing=5e-5, .film=5e-5: small penalties for using expensive features. MCP learns to prefer cheaper paths when accuracy allows.

### What actually happens at runtime
- Per input, MCP outputs gates 0..1 for each feature.
- Training: gates are soft; loss includes mcp_cost_total + your normal losses.
- Eval: gates are thresholded; features with gate OFF are fully skipped.
- Cycles/steps/MLP width are chosen per input within your ceilings.
- Ponder: higher ponder gate → stronger KL and earlier halting threshold; lower gate → lighter pondering.

This setup makes MCP the “effort controller” for CMBA: fewer steps and features on easy inputs, more on hard ones, all learned end-to-end.

Here's the complete list of MCP-controlled features in Document 6:

| Feature | Gate Name | Effect |
|---------|-----------|---------|
| Puzzle Embedding | `puzzle` | Blends between table lookup vs PQC embedding |
| Halting Decision | `halt` | Adds delta to halt/continue Q-values |
| Residual Gates | `gate` | Scales attention and MLP residual connections |
| Per-Head Attention | `headbias` | Scales importance of individual attention heads |
| Token Routing | `routing` | Selects subset of tokens to process |
| FiLM Conditioning | `film` | Applies feature-wise linear modulation |
| RoPE Phase | `rope` | Adjusts rotary position encoding phases |
| ACT Scheduler | `sched` | Controls H→L and L→H information flow gates |
| Ponder Weight | `ponder` | Modulates Ponder KL weight and eval halt threshold |
| NTM Memory | `ntm` | Gates external memory read/write operations |
| H-Level Cycles | `h_cycles` | Sets number of high-level reasoning iterations (1 to max) |
| L-Level Cycles | `l_cycles` | Sets number of low-level reasoning iterations (1 to max) |
| MLP Expansion | `mlp_expand` | Scales MLP width from 1x to max expansion |
| Active Heads | `heads_active` | Reserved for head-level gating (not implemented) |
| Minimum Steps | `min_steps` | Sets per-sample minimum allowed steps (0..ceiling) |
| Maximum Steps | `max_steps` | Sets per-sample maximum allowed steps (1..ceiling) |

Note: All gates output values in [0,1] range, with thresholding at `mcp_eval_threshold` (default 0.5) during evaluation when `mcp_hard_eval` is enabled.

Metrics to watch:
- `gate_h_cycles`, `gate_l_cycles`, `gate_mlp_expand`, `gate_ntm`, `gate_routing`, `gate_film`, etc.
- `mcp_cost_total`, and per-feature: `mcp_cost_ntm`, `mcp_cost_routing`, `mcp_cost_film`.

=================================================================================
### FAQ (short and clear)
=================================================================================
- Do I need all MCP flags? No. Required: `arch.mcp_enabled=true`. Recommended: `arch.mcp_hard_eval=true arch.mcp_auto_features=true`. Others have safe defaults.
- What do “ceilings” mean? They are upper limits; MCP picks the actual value per input (e.g., 1..3 cycles).
- Will this speed up eval? Yes. Features with gate OFF are skipped; cycles and steps shrink for easy inputs.
- Will accuracy drop? Usually not; MCP learns to spend more where needed. You can add small per-feature costs to encourage cheaper paths.
- Can I add new features later? Yes. Add a gate name in `mcp_feature_keys`, give it an optional cost, and scale/skip that block by the gate.

### Tips
- Start with the quick start block. If throughput is the priority, set `arch.mcp_feature_profile=fast` to bias MCP toward cheaper paths early on.
- If gates look too soft in training, reduce `arch.mcp_temp` (e.g., 0.5). If too hard, increase it.
- For more aggressive skipping at eval, set `arch.mcp_eval_threshold=0.6–0.7`.

=================================================================================
=================================================================================
=================================================================================
