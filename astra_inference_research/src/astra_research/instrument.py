"""Attach measurement and optional residual-cache hooks to official Astra DiT.

Verified against EternalEvan/Astra:
- infer_demo.inference_moe_framepack_sliding_window builds pipe.dit
- WanModelMoe.forward calls self.blocks sequentially
- official infer_demo sets scheduler.set_timesteps(50)

Hooks depend on CALL ORDER for step/branch indexing. Do not use cache mode
with a modified upstream sampling loop without updating this adapter.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any
import types

from .cache import ResidualCache
from .probe import ChunkProbe


@dataclass(frozen=True)
class CallContext:
    chunk: int
    step: int
    branch: int


class AstraInstrumentation:
    def __init__(self, config: dict[str, Any], cache_enabled: bool):
        import torch
        self.torch = torch
        self.config = config
        self.cache_config = config["cache"]
        self.cache = ResidualCache(
            enabled=cache_enabled,
            indices=[int(i) for i in self.cache_config.get("block_indices", [])],
            interval=int(self.cache_config.get("interval", 2)),
            max_age=int(self.cache_config.get("max_age", 1)),
            relative_threshold=self.cache_config.get("relative_threshold"),
        )
        probe_config = config.get("probe", {})
        self.probe = ChunkProbe(
            enabled=bool(probe_config.get("enabled", False)),
            block_indices=[int(i) for i in probe_config.get("block_indices", [])],
        )
        self.expected_steps = int(config["measurement"].get("expected_denoising_steps", 50))
        gen = config["generation"]
        if float(gen.get("text_guidance_scale", 1.0)) != 1.0:
            raise ValueError(
                "Instrumentation currently supports text_guidance_scale=1.0 only; "
                "additional CFG branches require verified step/branch mapping.")
        self.branches = 2 if (gen.get("use_camera_cfg", False) and
                              float(gen.get("camera_guidance_scale", 2.0)) > 1.0) else 1
        self.forward_calls = 0
        self.current: CallContext | None = None
        self.cuda_event_pairs: list[tuple[Any, Any, CallContext]] = []
        self.hooks = []
        self._installed = False
        self.block_call_counts: dict[int, int] = defaultdict(int)
        self._pipe = None

    def install(self, pipe: Any) -> None:
        if self._installed:
            raise RuntimeError("Instrumentation already installed")
        self._installed = True
        self._pipe = pipe
        dit = pipe.dit
        blocks = list(dit.blocks)
        needed = self.cache.indices | self.probe.blocks
        invalid = needed - set(range(len(blocks)))
        if invalid:
            raise ValueError(f"Invalid DiT block indices: {sorted(invalid)}; total={len(blocks)}")
        if self.cache.enabled and not self.cache.indices:
            raise ValueError("Cache enabled, but block_indices is empty; no actual optimization configured")

        # Pre/post hooks measure the entire DiT including input projection/heads.
        # No numerical changes are made when cache_enabled=False.
        self.hooks.append(dit.register_forward_pre_hook(self._dit_pre, with_kwargs=True))
        self.hooks.append(dit.register_forward_hook(self._dit_post, with_kwargs=True))
        for index in sorted(needed):
            block = blocks[index]
            original = block.forward
            instrumentation = self

            def make_wrapper(index: int, original: Any):
                def wrapper(self_block: Any, x: Any, *args: Any, **kwargs: Any):
                    ctx = instrumentation.current
                    if ctx is None:
                        return original(x, *args, **kwargs)
                    instrumentation.block_call_counts[index] += 1
                    result = instrumentation.cache.execute(
                        index, x, lambda: original(x, *args, **kwargs),
                        chunk=ctx.chunk, step=ctx.step, branch=ctx.branch,
                    )
                    instrumentation.probe.observe(
                        index, result, chunk=ctx.chunk, step=ctx.step, branch=ctx.branch)
                    return result
                return wrapper

            block.forward = types.MethodType(make_wrapper(index, original), block)

    def _dit_pre(self, module: Any, args: tuple, kwargs: dict):
        if len(self._pipe.scheduler.timesteps) != self.expected_steps:
            raise RuntimeError(
                f"Expected {self.expected_steps} scheduler steps; got "
                f"{len(self._pipe.scheduler.timesteps)}. "
                "Update expected_denoising_steps only after inspecting the official loop.")
        ordinal = self.forward_calls
        calls_per_chunk = self.expected_steps * self.branches
        ctx = CallContext(
            chunk=ordinal // calls_per_chunk,
            step=(ordinal % calls_per_chunk) // self.branches,
            branch=ordinal % self.branches,
        )
        self.forward_calls += 1
        self.current = ctx
        self.cache.begin_chunk(ctx.chunk)
        if self.torch.cuda.is_available():
            start = self.torch.cuda.Event(enable_timing=True)
            end = self.torch.cuda.Event(enable_timing=True)
            start.record()
            self.cuda_event_pairs.append((start, end, ctx))

    def _dit_post(self, module: Any, args: tuple, kwargs: dict, output: Any):
        if self.cuda_event_pairs:
            self.cuda_event_pairs[-1][1].record()

    def results(self) -> dict[str, Any]:
        torch = self.torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        per_chunk: dict[int, float] = defaultdict(float)
        per_step: dict[str, float] = defaultdict(float)
        for start, end, ctx in self.cuda_event_pairs:
            elapsed = float(start.elapsed_time(end))
            per_chunk[ctx.chunk] += elapsed
            per_step[f"{ctx.chunk}:{ctx.step}"] += elapsed
        return {
            "dit_forward_calls": self.forward_calls,
            "expected_steps_per_chunk": self.expected_steps,
            "assumed_cfg_branches_per_step": self.branches,
            "complete_chunks_by_call_count": self.forward_calls // (self.expected_steps * self.branches),
            "dit_cuda_time_ms_sum": sum(per_chunk.values()) if self.cuda_event_pairs else None,
            "dit_cuda_ms_by_chunk": dict(per_chunk),
            "dit_cuda_ms_by_chunk_step": dict(per_step),
            "selected_block_calls": dict(self.block_call_counts),
            "cache": self.cache.stats(),
            "cross_chunk_probe": self.probe.export(),
            "warning": (
                "CUDA events are instrumented timings; compare unprofiled end-to-end wall time "
                "for actual speedups. Step/branch mapping assumes official infer_demo call order."
            ),
        }
