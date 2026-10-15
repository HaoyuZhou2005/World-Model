"""Hooks for verified Astra token packing and query-sparse block execution.

Explicit experimental assumptions: 50-step original infer_demo, one/two
camera CFG branches (not text CFG), one sample, original WanModelMoe.blocks.
We do NOT patch cross-chunk token mapping or modify Astra checkpoint.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any
import types

from .gate import GateConfig, TokenGate
from .layout import LayoutRecorder
from .token_cache import TokenResidualCache
from .collector import TeacherCollector


@dataclass(frozen=True)
class Context:
    chunk: int
    step: int
    branch: int


class TokenInstrumentation:
    def __init__(self, settings: dict, cache_enabled: bool):
        import torch
        self.torch = torch
        self.settings = settings
        gen = settings["generation"]
        if float(gen.get("text_guidance_scale", 1.0)) != 1.0:
            raise ValueError("Unsupported text CFG: requires separately verified forward ordering")
        self.steps = int(settings["measurement"].get("steps", 50))
        if self.steps != 50:
            raise ValueError("Official Astra infer_demo currently hardcodes 50 denoising steps")
        self.branches = (2 if gen.get("use_camera_cfg", False) and
                         float(gen.get("camera_guidance_scale", 2.0)) > 1 else 1)
        cache_cfg = settings["token_cache"]
        gate_config = GateConfig(
            ratios=cache_cfg["ratios"],
            coarse_threshold=cache_cfg.get("coarse_threshold"),
            min_per_stream=int(cache_cfg.get("min_per_stream",1)),
            max_age=int(cache_cfg.get("max_age",1)),
            first_full_steps=int(cache_cfg.get("first_full_steps",1)),
            force_last_step_full=bool(cache_cfg.get("force_last_step_full",True)),
            horizon_boost=float(cache_cfg.get("horizon_boost",0.0)))
        self.token_cache = TokenResidualCache(
            TokenGate(gate_config, self.steps,
                      policy=("change_topk" if cache_cfg.get("policy")=="intervention"
                              else cache_cfg.get("policy","change_topk")),
                      model_checkpoint=cache_cfg.get("model_checkpoint")),
            selected_blocks=[int(i) for i in cache_cfg.get("blocks",[])],
            enabled=cache_enabled, intervention=cache_cfg.get("intervention"))
        if cache_enabled and not self.token_cache.selected_blocks:
            raise ValueError("No selected blocks; no optimization would occur")
        collection=settings.get("collection",{})
        self.collector=TeacherCollector(int(collection.get("samples_per_step",128))) if collection.get("enabled",False) else None
        if self.collector is not None and cache_enabled:
            raise ValueError("Teacher collection requires NO-cache full model computations")
        self.recorder = LayoutRecorder()
        self.context: Context | None = None
        self.forward_count = 0
        self.pre_and_post_events = []
        self.cuda_events = []
        self.layouts_by_chunk: dict[int, dict] = {}
        self.blocks_wrapped = 0
        self._pipe = None
        self._handles = []

    def _before_dit(self, _model, args, kwargs):
        if len(self._pipe.scheduler.timesteps) != self.steps:
            raise RuntimeError("Astra scheduler changed; abort rather than corrupt step index")
        ordinal = self.forward_count
        per_chunk = self.steps * self.branches
        self.context = Context(
            chunk=ordinal // per_chunk,
            step=(ordinal % per_chunk) // self.branches,
            branch=ordinal % self.branches)
        self.forward_count += 1
        self.recorder.reset()
        self.token_cache.begin_chunk(self.context.chunk)
        if self.torch.cuda.is_available():
            start = self.torch.cuda.Event(enable_timing=True)
            end = self.torch.cuda.Event(enable_timing=True)
            start.record()
            self.cuda_events.append((start, end, self.context))

    def _after_dit(self, _model, args, kwargs, output):
        if self.cuda_events:
            self.cuda_events[-1][1].record()

    def install(self, pipe) -> None:
        if self._pipe is not None:
            raise RuntimeError("Already installed")
        self._pipe = pipe
        dit = pipe.dit
        if not hasattr(dit, "patch_embedding") or not hasattr(dit,"clean_x_embedder"):
            raise RuntimeError("Expected WanModelMoe patch_embedding + FramePack clean_x_embedder")
        blocks = list(dit.blocks)
        invalid = self.token_cache.selected_blocks - set(range(len(blocks)))
        if invalid:
            raise ValueError(f"Nonexistent block indices: {sorted(invalid)}, total {len(blocks)}")
        self._handles.append(dit.register_forward_pre_hook(self._before_dit, with_kwargs=True))
        self._handles.append(dit.register_forward_hook(self._after_dit, with_kwargs=True))
        self._handles.append(dit.patch_embedding.register_forward_hook(
            self.recorder.capture_target))
        self._handles.append(dit.clean_x_embedder.register_forward_hook(
            self.recorder.capture_history, with_kwargs=True))
        for block_id in sorted(self.token_cache.selected_blocks):
            block = blocks[block_id]
            original = block.forward
            outer = self
            def make_wrapper(block_id, original):
                def forward(self_block, x, *call_args, **call_kwargs):
                    if outer.context is None:
                        raise RuntimeError("Block execution without DiT context")
                    layout = outer.recorder.layout(x.shape[1])
                    context = outer.context
                    if context.step == 0 and context.branch == 0:
                        outer.layouts_by_chunk[context.chunk] = layout.as_dict()
                    if outer.collector is not None:
                        full = original(x, *call_args, **call_kwargs)
                        outer.collector.observe(x, full, layout=layout,
                            chunk=context.chunk, step=context.step, branch=context.branch,
                            block_id=block_id, steps=outer.steps)
                        return full
                    return outer.token_cache.run(
                        self_block, block_id, x, original, call_args, call_kwargs,
                        layout=layout, chunk=context.chunk,
                        step=context.step, branch=context.branch)
                return forward
            block.forward = types.MethodType(make_wrapper(block_id, original), block)
            self.blocks_wrapped += 1

    def results(self):
        torch = self.torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        by_chunk = defaultdict(float)
        for a,b,c in self.cuda_events:
            by_chunk[c.chunk] += float(a.elapsed_time(b))
        return {
            "dit_forward_calls": self.forward_count,
            "assumed_denoising_steps": self.steps,
            "branches_per_step": self.branches,
            "full_chunks": self.forward_count // (self.steps*self.branches),
            "completed_step_alignment": self.forward_count % (self.steps*self.branches) == 0,
            "dit_cuda_ms_by_chunk": dict(by_chunk),
            "dit_cuda_ms_total": float(sum(by_chunk.values())) if self.cuda_events else None,
            "layout_by_chunk": self.layouts_by_chunk,
            "cache": self.token_cache.statistics(),
            "teacher_records": self.collector.rows if self.collector is not None else 0,
            "caveat": "CUDA events include hook overhead; skipped queries != skipped model FLOPs",
        }
