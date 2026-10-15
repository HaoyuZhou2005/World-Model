# 101601 — Proposal-aligned Token-level AdaptiveCache

**Core change:** Unlike 101501 whole-block residual scheduling, **101601 selects
fresh vs stale computations at TOKEN / FramePack temporal-frame-group level**.

Research basis: the user's seven-page 2026-09-08 proposal and 2026-10-16
101501 experiment feedback. Read [Research Plan](docs/RESEARCH_PLAN.md)
before claiming any results.

## Scientific / engineering status

| Component | Status |
| --- | --- |
| Recover actual [4x \| 2x \| 1x \| target] token offsets from official Astra Conv3D projection | Code written |
| Per-token reuse age / cache state / CFG-branch isolation / clear at each new chunk | Code written |
| Sparse Q rows with FULL K/V, selected-row attention/cross-attention/FFN | Prototype written; official GPU numerics unverified |
| Coarse temporal frame-group feature-change proxy + per-token top-k | Prototype written |
| Short-horizon (step/age/change) and long-horizon (chunk-conditioned budget) inputs | Rule baseline only |
| Local residual teacher dataset and small MLP | Code written; NOT future-safe |
| Explicit one-token stale intervention and future rollout proxy labels | Code written; GPU runs required |
| Learned future-risk importance from measured interventions | Training code written; dataset NOT generated |
| Cross-AR-chunk hidden/KV feature reuse | **NOT IMPLEMENTED** |
| End-to-end differentiable multi-chunk gate training | **NOT IMPLEMENTED** |
| Full Astra GPU test or speedup evidence | **NOT EXECUTED** |

**No model retraining** is needed for the baseline/sparse experiments; only
the optional small gate MLP is trained offline.

## Layout and real computation

Upstream checked: EternalEvan/Astra — WanModelMoe.process_input_hidden_states.
Packing is [4x history | 2x history | 1x history | noisy target].

A fully dense block produces residual d = block(x) - x for all tokens.
At later steps, for selected set I:
- compute Q and attention output for query rows I;
- K/V projections still cover all source tokens;
- cross-attention and FFN output only for I;
- tokens outside I return x + previous per-token cached residual;
- no crossing chunk boundaries, and CFG conditional/unconditional caches separate.

This is **not** the false optimization of calculating a full block and
merely replacing some outputs with a mask. Still, the current prototype has
full K/V/MoE costs and dynamic indexing overhead; speedup may be negative.

## Project structure

    src/astra_token_research/
      layout.py           Observe actual FramePack token boundaries
      gate.py             Group coarse screen and token top-k, or trained MLP
      sparse_block.py     Selected-query Astra block compute
      token_cache.py      Token residual lifecycle, optional one-token intervention
      instrument.py       Instrument official DiT and its 50-step call sequence
      runner.py           Run official inference, unchanged model checkpoint
      safety.py           Single idle GPU guard for shared server
      features.py         Short/long conditioned token features (9D schema)
      collector.py        Full-teacher local residual error dataset
      future_safety.py    Future chunk distillation objective definitions
      counterfactual.py   Measured one-token multi-chunk future-risk dataset
      oracle_plan.py      Generate low-count token intervention configs
      train_gate.py       Local vs future oracle supervised gate MLP
      cli.py              Preflight / baseline / token / teacher / gate training

Config: configs/proposal_token.json
Tests: tests/

## Basic commands

First install official Astra and checkpoint on the GPU server as you already
did for 101501. This repository DOES NOT bundle model weights or modify them.

    cd /YOUR/World-Model/astra_inference_research/101601
    python -m pip install -e ".[test]"
    python -m pytest -q tests
    cp configs/proposal_token.json configs/local.json

Fill all TODO paths and the **actual successful garden_1 prompt** in local.json.

    python -m astra_token_research.cli run --config configs/local.json --mode baseline --dry-run
    python -m astra_token_research.cli doctor --config configs/local.json

Manually inspect free GPU memory, then choose one physical GPU:

    nvidia-smi
    export CUDA_VISIBLE_DEVICES=YOUR_IDLE_GPU_INDEX
    export PYTHONPATH=$PWD/src
    python -m astra_token_research.cli run --config configs/local.json --mode baseline --out runs
    python -m astra_token_research.cli run --config configs/local.json --mode token --out runs

Teacher collection and one-token counterfactual procedure:
[Execution Guide](docs/EXECUTION.md).

## Acceptance criteria

1. Original official Astra output vs 101601 no-cache must match at fixed seeds.
2. Selected rows from sparse_block must match the original full block on the
   SAME input for REAL model, RoPE, MoAE and camera conditions (FP tolerance).
3. Token choice must reduce actually computed Q/attention/FFN rows, not just
   mask the result; measure FLOPs and actual CUDA kernel latency separately.
4. Benchmark across seeds and scenarios; include ≥3 autoregressive chunks,
   measure future motion/camera/structure integrity, not only MAE vs baseline.
5. K4 whole-block 101501 is an independent baseline. Failure to beat K4
   does NOT prevent token-gate mechanism development.
6. Cross-chunk reuse requires a separate validated token identity map.

See [Research Plan](docs/RESEARCH_PLAN.md) for exact original-proposal
correspondence and research limitations.
