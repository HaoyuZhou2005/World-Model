# Astra integration audit and unresolved items

## Confirmed upstream implementation

Source: https://github.com/EternalEvan/Astra (ICLR 2026), inspected on 2026-10-14.
Clone and record an exact git rev-parse HEAD in every result. Upstream may change.

| Item | Verified location | Integration decision |
| --- | --- | --- |
| Official inference | scripts/infer_demo.py | Call inference_moe_framepack_sliding_window directly; do not recreate AR rollout |
| Pipeline | diffsynth/pipelines/wan_video_recammaster.py | WanVideoAstraPipeline class |
| DiT | diffsynth/models/wan_video_dit_moe.py | WanModelMoe, DiTBlockWithMoE |
| Blocks | WanModelMoe.blocks | Opt-in forward wrapper around selected blocks |
| Denoising | inference_moe_framepack_sliding_window | pipe.scheduler.set_timesteps(50) and pipe.dit per step |
| AR history | Same function | history_latents = cat(history_latents, new_latents) after each chunk |
| FramePack | prepare_framepack_sliding_window_with_camera_moe | Leave unchanged |
| Camera guidance | Conditional/unconditional branches | Partition caches by branch index |
| DiT return | WanModelMoe.forward | Tuple of predicted latent/noise and MoE loss |
| Original checkpoint | models/Astra/checkpoints/diffusion_pytorch_model.ckpt | Load without training |
| Wan files | Wan2.1-T2V-1.3B folder | Check all three expected model files |

Official CLI marks condition_image required even though the inner function supports
pre-encoded condition data. This research adapter requires a valid image and lets
condition_pth override it. This is a compatibility compromise, not a model property.
The AR chunk counts in the example use latent frames, not RGB video frames.

## TODO: local information to fill in

- **TODO_ASTRA_REPO**: absolute path to a clean clone of EternalEvan/Astra.
- **TODO_ASTRA_CKPT**: absolute path to the released Astra checkpoint.
- **TODO_WAN_MODEL_DIR**: absolute path to the Wan2.1-T2V-1.3B model directory.
- **TODO_CONDITION_IMAGE**: local PNG/JPEG input.
- **TODO_CONDITION_PTH**: optional preprocessed latents and camera data; check schema.
- **TODO_EVAL_DATASET**: benchmark split, camera trajectories, GT and license.
- **TODO_SERVER_GPU**: CUDA/PyTorch, GPU model, VRAM, official Astra installation.
- **TODO_UPSTREAM_PIN**: pin upstream commit and validate checkpoint compatibility.
- **TODO_METRICS**: standard generative video/world-model metrics after dataset alignment.

**No placeholders are silently replaced with fabricated local paths.**

## Scientific limitations and safeguards

1. Residual caching is only within one AR chunk. It is approximate, not a verified technique.
2. All residuals clear at chunk boundaries. Future inter-chunk reuse needs verified
   token correspondence across FramePack 1x/2x/4x streams and camera/time positions.
3. ChunkProbe reports global pooled feature cosine at the first denoising step.
   This is **not** token-level spatial or temporal correspondence.
4. Guidance branch indexing uses official call order. The script uses 50 steps,
   one forward per step without guidance, two with camera guidance. Text guidance
   greater than 1 is deliberately rejected.
5. The optional relative-threshold check adds GPU/CPU sync overhead.
6. CUDA events measure DiT forward time. Wall clock also includes checkpoint
   loading, VAE, decoding, Python and video I/O. Report them separately.
7. Torch profiler tracing adds substantial overhead; do not compare traced runs.
8. GPU execution and Astra checkpoint loading have NOT been tested here.
9. Upstream uses non-strict checkpoint loading. Unexpected/missing keys need to
   be inspected for accurate reproduction; document their count before publication.
10. Cross-chunk caching is NOT implemented. Long-horizon feature pooling is only
    an observational analysis, not evidence of safe reuse.
11. The hooks use an exact expected-step count of 50. A changed upstream sampler
    must force adapter review, not silently produce misleading cache decisions.
12. For final baseline fairness, verify that no-cache instrumentation preserves
    output numerics compared to an unmodified official Astra inference command.

## Follow-up development

Stage A: verify official checkpoint and fixed-seed no-cache equivalence with upstream.
Stage B: measure per-layer change within the same chunk, schedule-step distances,
runtime overhead and quality sensitivity for fixed residual reuse.
Stage C: develop an explicit token-index map for overlapping history inputs across
chunks, including RoPE, FramePack scales, camera actions, MoAE and CFG.
Stage D: validate cross-chunk reuse as its own ablation. Only after a reproducible
mechanism exists should a dual-horizon controller or trainable gate be added.

Astra No Cache vs Astra Cache must use the same weights and sampler. For a sound
speedup table, include warmup, repeated trials, medians/dispersion, identical
seeds, benchmark subsets and quality metrics that measure rollout stability.
