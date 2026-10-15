# Astra Inference Optimization Research

**Research scaffold, not a validated acceleration result.** This directory lives independently of the existing World-Model models/ and note/ folders.

Goal: use the **published ICLR 2026 Astra** model and released Wan2.1-based checkpoint, without training, to study **denoising-step redundancy**, approximate **residual reuse inside a chunk**, and **cross-chunk feature similarity**. The official Astra AR rollout, FramePack history, camera conditioning, MoAE, and decoding remain in place.

Official source: https://github.com/EternalEvan/Astra
Official weights: https://huggingface.co/EvanEternal/Astra
Upstream inference entrypoint: Astra/scripts/infer_demo.py

## What works in source code

| Function | State |
| --- | --- |
| Configuration validation; no fabricated local paths | Implemented |
| Original Astra inference invoked in-process | Adapter implemented, GPU run NOT verified |
| No-cache baseline with forward counts and CUDA events | Implemented, GPU run NOT verified |
| Opt-in fixed-interval DiT block residual reuse | Research prototype, NOT quality validated |
| Per-branch cache isolation and clearing between AR chunks | Implemented |
| Pooled activation similarities across steps/chunks (opt-in) | Analysis only, NOT token alignment |
| JSON experiment metadata; elapsed time; peak GPU memory | Implemented |
| Paired video pixel deviation and run comparison | Implemented |
| Cross-chunk cache with positional token alignment | **TODO / NOT IMPLEMENTED** |
| Learned token/frame gate, trained policy | **TODO / NOT IMPLEMENTED** |
| Benchmark quality metrics (FVD, VBench, camera pose, etc.) | **TODO / NOT IMPLEMENTED** |

Do not report expected speedups or FLOPs. Model loading, VAE, decoding and profiling hooks can change wall time. The residual method deliberately changes DiT predictions; it needs full AR rollout quality evaluation.

## Directory structure

~~~text
astra_inference_research/
├── README.md
├── .gitignore
├── pyproject.toml
├── configs/example.json
├── docs/INTEGRATION.md
├── src/astra_research/
│   ├── __init__.py
│   ├── config.py       # fail-closed config, path preflight
│   ├── adapter.py      # dynamically call official Astra inference
│   ├── instrument.py   # DiT/block hooks, timing, call-order mapping
│   ├── cache.py        # approximate denoising-step residual cache
│   ├── probe.py        # pooled feature statistics across AR chunks
│   ├── report.py       # saved-run comparison and video deviation
│   └── cli.py          # doctor / run / compare
└── tests/
    ├── test_cache.py
    └── test_config.py
~~~

## Install and configure

A CUDA Linux environment with compatible PyTorch is expected. The official repository says a 24 GB GPU (such as RTX 3090) can run its example. Hardware compatibility is not verified by this scaffold.

~~~bash
git clone https://github.com/EternalEvan/Astra.git /YOUR/Astra
cd /YOUR/Astra
pip install -e .

cd /YOUR/World-Model/astra_inference_research
pip install -e ".[dev]"
cp configs/example.json configs/local.json
~~~

Edit only your own configs/local.json (ignored by Git). Replace these placeholders:

- TODO_ABSOLUTE_PATH_TO_OFFICIAL_ASTRA_CLONE
- TODO_ABSOLUTE_PATH_TO_ASTRA_DIFFUSION_PYTORCH_MODEL_CKPT
- TODO_ABSOLUTE_PATH_TO_WAN2_1_T2V_1_3B
- TODO_ABSOLUTE_PATH_TO_CONDITION_IMAGE_PNG

The example has 50 denoising steps, 4 latent frames per chunk, 8 latent generated frames total, 3 MoE experts/Top-K 1, and camera type 4. These are initial study settings based on the checked official inference script, **not published benchmark settings**. Match the official checkpoint and your benchmark configs before serious experiments. Never use an unverified checkpoint with a guessed MoE layout.

## Run

~~~bash
# Works without checking for a GPU; prints configured values/placeholders.
python -m astra_research.cli run --config configs/example.json --mode baseline --dry-run

# Verifies your real paths, without loading model weights.
python -m astra_research.cli doctor --config configs/local.json

# Vanilla Astra (feature cache disabled); saves generated.mp4 and metadata.json.
python -m astra_research.cli run --config configs/local.json --mode baseline

# Experimental within-chunk residual reuse, SAME model and inputs.
python -m astra_research.cli run --config configs/local.json --mode cache

# Compare the two generated run folders.
python -m astra_research.cli compare \
  --reference runs/BASELINE_RUN_FOLDER \
  --candidate runs/CACHE_RUN_FOLDER \
  --video-metrics

# Optional slow profiler trace; NEVER use traced times for speedup tables.
python -m astra_research.cli run --config configs/local.json --mode baseline --profiler
~~~

Note: output_root in the provided JSON is ../runs relative to configs/, so the resulting location is astra_inference_research/runs.

For a pristine, uninstrumented reference you can run the official script with the **same config parameters** from Astra/scripts/infer_demo.py and compare the output. The baseline in this directory attaches no-cache hooks for measurement and should be checked against that unmodified output before research claims.

## Cache design and limitations

The first approximation is residual reuse:

~~~text
full block:    residual = block(x) - x
reused block:  output   = x + cached_residual
~~~

At denoising steps divisible by interval, the chosen blocks execute normally and refresh the residual. Other steps may skip the block under max_age and shape/dtype/device checks. The cache is indexed by (CFG branch, block index) and cleared at every new chunk. Setting relative_threshold enables an input-change gate but causes CPU/GPU synchronization. All thresholds and block choices are examples, not tuned results.

This is **not** a free lunch. Caching residuals consumes GPU memory. Large quality changes may make it unsuitable. The first safe experiment is to run a single selected block with interval 2 and compare both runtime and generated output. The baseline must run with caching disabled.

For intra-step research, enable probe.enabled and probe.step_analysis in a separate baseline (no-cache) diagnostic run. Disable probes for both runtime-comparison runs. The probe also records pooled hidden vectors at the first denoising step across chunks. These are coarse descriptors; they **do not** align FramePack token positions. A high cosine similarity cannot justify direct cross-chunk token reuse. Implement cross-chunk acceleration only after mapping history tokens across chunks.

## Minimal experiment protocol

- Pin an exact Astra Git commit and checksum the Wan/Astra weights.
- Verify official example inference first, then our no-cache wrapper.
- Run identical seeds, prompt, camera trajectory, scheduler, resolution, chunk length and checkpoint.
- Record CUDA sync wall time, DiT event time, VRAM and forward count; do NOT call skipped blocks "FLOPs saved" without correct computation accounting.
- Test long-horizon video quality and camera control, not just pixel MAE vs baseline.
- Repeat after warmup and report median, spread, and all hardware/driver versions.
- Benchmark a single-chunk and a multi-chunk setting; check regression/compounding errors.
- Save the exact JSON, SHA, run metadata and original/cached MP4.

## Tests

~~~bash
pytest -q tests/
~~~

These are **CPU synthetic tests** of bookkeeping and hooks. They cannot establish that the official checkpoint loads or that GPU outputs are correct.

## Follow-up research

See docs/INTEGRATION.md for exact verified upstream class/function locations, unresolved environment placeholders, current assumptions and required experiments. This stage does **not** modify or retrain Astra, and no server data, benchmark samples, or checkpoint binaries are bundled.
