"""Approximate DiT block RESIDUAL reuse across denoising steps.

WARNING: experimental / quality not validated. This is NOT an attention KV cache.
It never shares residuals between CFG branches or autoregressive chunks.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class CacheEntry:
    residual: Any
    input_tensor: Any
    last_full_step: int
    shape: tuple[int, ...]
    dtype: Any
    device: Any


class ResidualCache:
    def __init__(self, enabled: bool, indices: list[int], interval: int = 2,
                 max_age: int = 1, relative_threshold: float | None = None):
        if interval < 2 or max_age < 1:
            raise ValueError("interval >= 2 and max_age >= 1 required")
        self.enabled = enabled
        self.indices = set(indices)
        self.interval = interval
        self.max_age = max_age
        self.relative_threshold = relative_threshold
        self.entries: dict[tuple[int, int], CacheEntry] = {}
        self.chunk = -1
        self.reused = 0
        self.recomputed = 0
        self.threshold_rejected = 0
        self.reused_by_block: dict[int, int] = {}

    def begin_chunk(self, chunk: int) -> None:
        if chunk != self.chunk:
            # Cross-chunk token positions, history and FramePack layout differ.
            # Never reuse a residual blindly at this boundary.
            self.entries.clear()
            self.chunk = chunk

    def execute(self, block_index: int, x: Any, forward_fn: Any, *,
                chunk: int, step: int, branch: int) -> Any:
        if block_index not in self.indices:
            return forward_fn()
        self.begin_chunk(chunk)
        key = (branch, block_index)
        entry = self.entries.get(key)
        eligible = (
            self.enabled and step > 0 and step % self.interval != 0
            and entry is not None and step > entry.last_full_step
            and step - entry.last_full_step <= self.max_age
            and tuple(x.shape) == entry.shape
            and x.dtype == entry.dtype and x.device == entry.device
        )
        if eligible and self.relative_threshold is not None:
            # Explicit GPU sync here makes threshold-gated runs slower.
            # Measure this overhead; do not hide it in reported speedups.
            numerator = (x.detach().float() - entry.input_tensor.float()).abs().mean()
            denominator = entry.input_tensor.float().abs().mean().clamp_min(1e-6)
            eligible = bool((numerator / denominator <= self.relative_threshold).item())
            if not eligible:
                self.threshold_rejected += 1
        if eligible:
            self.reused += 1
            self.reused_by_block[block_index] = self.reused_by_block.get(block_index, 0) + 1
            return x + entry.residual

        result = forward_fn()
        self.recomputed += 1
        if self.enabled:
            # Save the block's residual rather than a stale absolute output.
            self.entries[key] = CacheEntry(
                residual=(result - x).detach(),
                input_tensor=x.detach() if self.relative_threshold is not None else None,
                last_full_step=step, shape=tuple(x.shape), dtype=x.dtype, device=x.device,
            )
        return result

    def stats(self) -> dict[str, Any]:
        total = self.reused + self.recomputed
        return {
            "enabled": self.enabled, "selected_blocks": sorted(self.indices),
            "reused_block_calls": self.reused,
            "full_block_calls": self.recomputed,
            "reuse_ratio_selected_calls": self.reused / total if total else 0.0,
            "threshold_rejected": self.threshold_rejected,
            "reused_by_block": self.reused_by_block,
            "remark": "Skipped block calls, NOT verified FLOPs or speedup.",
        }
