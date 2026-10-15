# 101501 | Astra inference optimization — audited continuation

**Status:** Code implemented for the next experimental phase; GPU and official-checkpoint integration tests NOT run in this session. No new model training.

- Original GitHub scaffold is preserved in **[../101401/](../101401/)**.
- This is **101501**, based on an audit of the uploaded `26101501.zip`.
- Audit: **[docs/AUDIT.md](docs/AUDIT.md)**.
- Detailed local commands and validation plan: **[docs/NEXT_EXPERIMENTS.md](docs/NEXT_EXPERIMENTS.md)**.

**Data availability clarification (from the user):** The experiments **did generate video files**, but those MP4s were **intentionally excluded from the uploaded ZIP to make it smaller**. The ZIP contains 86 completed `metadata.json` files and no MP4s. The recorded video-quality metrics therefore remain historical results that could not be independently recomputed from the supplied archive; retrieve the existing videos when independent visual/metric checks are needed.

## Scientific starting point

The upload reports (garden_1, seed 0, 3 × 8 latent frames, 50 denoise steps):

| Policy | DiT time | DiT speedup | MAE vs no-cache | MAE over chunks |
| --- | --- | --- | --- | --- |
| No cache | 138.6 s | 1× | — | — |
| Whole-block residual interval-2 | 71.2 s | 1.95× | 10.40 | 5.59 / 11.20 / 14.63 |
| interval-2 + first 2 steps full | 73.9 s | 1.88× | 8.95 | 5.89 / 9.08 / 12.06 |
| interval-2 + first 4 steps full | 76.8 s | 1.80× | 8.10 | 5.24 / 7.51 / 11.72 |

These are **DiT CUDA event times**, not end-to-end inference speedups.
MAE is to a no-cache *generated* video, not to ground truth.
These numbers are **historical reports**, not measurements of 101501 code.

**New hypothesis:** vary block refresh periods and chunk-boundary refresh windows independently; test if a Pareto improvement over uniform interval-2+K4 exists.

## Implemented modules

| Source | Function |
| --- | --- |
| src/astra_research/scheduled_cache.py | Per-block refresh period, chunk-specific K, fail-closed cache reuse |
| src/astra_research/instrument.py | Official WanModelMoe block instrumentation; no model weights changed |
| src/astra_research/experiment_plan.py | Ablation matrix generator |
| src/astra_research/benchmark.py | Strict matched repeated-run benchmarking |
| src/astra_research/evaluate.py | Exact-frame video output deviation by chunk |
| src/astra_research/identity.py | One-time SHA256 manifest for Astra source and all four model files |
| src/astra_research/study.py | Offline CLI for config matrix, pair analysis, video diagnostics |
| src/astra_research/adapter.py | Official Astra process adapter (from 101401) |
| tests/ | Synthetic CPU tests; do not prove CUDA correctness |

No frame/token learned gate and no valid cross-chunk feature reuse have been added.

## Install in your configured GPU server

~~~bash
cd /YOUR/World-Model/astra_inference_research/101501
python -m pip install -e ".[dev]"
python -m pytest -q tests/
cp configs/scheduled_example.json configs/local.json
~~~

1. Fill `configs/local.json` paths and the original verified garden prompt.
2. Hash the official model files **once** before running, with the commands below.
3. Put that manifest path in `paths.identity_manifest` (required by doctor/inference).

~~~bash
python -m astra_research.identity \
  --astra-repo /TODO/Astra \
  --astra-ckpt /TODO/Astra/checkpoints/diffusion_pytorch_model.ckpt \
  --wan-dir /TODO/Wan2.1-T2V-1.3B \
  --out configs/identity_local.json

python -m astra_research.cli doctor --config configs/local.json
python -m astra_research.study configs \
  --template configs/local.json --out configs/sweep_local --num-blocks 30
~~~

**Verify `len(pipe.dit.blocks)` equals 30** for your exact checkpoint before supplying num-blocks. The prior experiment used 30; do not generalize the number to other models.

On the shared GPU server, inspect `nvidia-smi`, choose *one free GPU*, and set `CUDA_VISIBLE_DEVICES` yourself. Never stop someone else's process or occupy all eight cards.

~~~bash
export CUDA_VISIBLE_DEVICES=TODO_IDLE_GPU
export PYTHONPATH=$PWD/src
python -m astra_research.cli run \
 --config configs/sweep_local/baseline.json --mode baseline

python -m astra_research.cli run \
 --config configs/sweep_local/uniform_i2_k4.json --mode cache
~~~

These are templates. You must replace TODO values; the scripts were not run on your GPU.

## Strict video diagnostics

After retrieving two actual `generated.mp4` files:

~~~bash
python -m astra_research.study videos \
 --reference /YOUR/BASELINE/generated.mp4 \
 --candidate /YOUR/CACHE/generated.mp4 \
 --rgb-per-chunk 32 --chunks 3 --condition-frames 1
~~~

The 97-frame layout is specific to the prior run. The analyzer refuses mismatched lengths and sizes instead of silently truncating.

## Strict repeated timing analysis

~~~bash
python -m astra_research.study pairs \
 --baseline /PATH/b1 /PATH/b2 /PATH/b3 \
 --cache /PATH/c1 /PATH/c2 /PATH/c3
~~~

Paired runs must have identical seed, model/input configuration, source evidence and weight manifests. Results separately report DiT speedup and end-to-end speedup. Multiple scenes and repetitions are required before publication.

**Critical limits:** A cached residual skips a DiT block inside one chunk; every chunk clears the cache. K=4 is *boundary re-computation*, not reuse across chunks. The current policy is deterministic/rule-based, not a trained adaptive gate. The newly written policies are experimental until official no-cache equivalence and CUDA quality tests pass.


## Optional ABBA sequential runner (single GPU; never trains)

The driver is **dry-run by default**. Use it only after the previous
identity/source and official-baseline equivalence checks have passed.

~~~bash
# No GPU action: print 2 cycles of baseline → cache → cache → baseline.
python -m astra_research.abba \
 --manifest configs/sweep_local/manifest.json \
 --baseline baseline --cache uniform_i2_k4 \
 --cycles 2 --logs /YOUR/DATA6/tmp/abba_101501

# Run expensive inference only after manual nvidia-smi inspection.
# export CUDA_VISIBLE_DEVICES=THE_IDLE_PHYSICAL_GPU
python -m astra_research.abba \
 --manifest configs/sweep_local/manifest.json \
 --baseline baseline --cache uniform_i2_k4 \
 --cycles 2 --logs /YOUR/DATA6/tmp/abba_101501 \
 --execute

python -m astra_research.abba_report \
 --manifest /YOUR/DATA6/tmp/abba_101501/abba_results.json \
 --out /YOUR/DATA6/tmp/abba_101501/summary.json
~~~

The runner executes sequentially (not concurrently), stops on the first
failure, and stores stdout/stderr per run. The report rejects mismatched
source code, weights, generation parameters, seeds, profiles or GPU identities.
**Do not run automatically on a shared machine unless explicitly intended.**
