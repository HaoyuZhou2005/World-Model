"""Training-free layer-aware intra-chunk residual reuse.

This implementation deliberately does NOT carry features across chunks:
FramePack history streams, RoPE indices and fresh target tokens change.

Schedule:
- every chunk's first \`warmup_steps\` are computed in full;
- a selected block with period p refreshes at steps s satisfying
  (s - warmup_steps) % p == 0; all other steps may reuse its residual;
- max_age bounds how far a residual can be reused after last full compute.
- p=1 means always recompute (use this for protected/sensitive layers).

No learned gate, no L1 threshold, no cross-chunk reuse, no quality guarantees.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping


@dataclass
class Entry:
    delta: Any
    full_step: int
    shape: tuple[int, ...]
    dtype: Any
    device: Any


class ScheduledResidualCache:
    def __init__(self, *, enabled: bool, indices: list[int],
                 period_by_block: Mapping[int | str, int],
                 default_period: int = 2,
                 warmup_steps: int = 1, max_age: int = 1,
                 warmup_by_chunk: Mapping[int | str, int] | None = None):
        self.enabled = bool(enabled)
        self.indices = set(int(i) for i in indices)
        self.period_by_block = {int(k): int(v) for k, v in period_by_block.items()}
        self.default_period = int(default_period)
        self.warmup_steps = int(warmup_steps)
        self.warmup_by_chunk = {int(k): int(v) for k, v in (warmup_by_chunk or {}).items()}
        self.max_age = int(max_age)
        if not self.indices and self.enabled:
            raise ValueError("Select at least one block for scheduled caching")
        if any(i < 0 for i in self.indices):
            raise ValueError("block indices must be nonnegative")
        if self.default_period < 1 or any(p < 1 for p in self.period_by_block.values()):
            raise ValueError("All periods must be >= 1; use 1 for full computation")
        if self.warmup_steps < 1 or any(w < 1 for w in self.warmup_by_chunk.values()):
            raise ValueError("At least step 0 must be computed in each chunk")
        if self.max_age < 1:
            raise ValueError("max_age must be >= 1")
        if set(self.period_by_block) - self.indices:
            raise ValueError("Period override for a block not selected for caching")
        self.entries: dict[tuple[int, int], Entry] = {}
        self.chunk: int | None = None
        self.reused = 0
        self.recomputed = 0
        self.reused_by_block: dict[int, int] = {}
        self.recomputed_by_block: dict[int, int] = {}
        self.skipped_in_warmup = 0

    def begin_chunk(self, chunk: int) -> None:
        if self.chunk != chunk:
            self.entries.clear()
            self.chunk = int(chunk)

    def _must_refresh(self, block: int, chunk: int, step: int) -> bool:
        warmup = self.warmup_by_chunk.get(chunk, self.warmup_steps)
        if step < warmup:
            return True
        period = self.period_by_block.get(block, self.default_period)
        return period == 1 or (step - warmup) % period == 0

    def execute(self, block_index: int, x: Any, forward_fn: Callable[[], Any], *,
                chunk: int, step: int, branch: int) -> Any:
        if not self.enabled or block_index not in self.indices:
            return forward_fn()
        if step < 0 or chunk < 0 or branch < 0:
            raise ValueError("Invalid chunk/step/branch")
        self.begin_chunk(chunk)
        key = (branch, block_index)
        entry = self.entries.get(key)
        must_refresh = self._must_refresh(block_index, chunk, step)
        can_reuse = (
            not must_refresh and entry is not None
            and 0 < step - entry.full_step <= self.max_age
            and tuple(x.shape) == entry.shape
            and x.dtype == entry.dtype and x.device == entry.device
        )
        if can_reuse:
            self.reused += 1
            self.reused_by_block[block_index] = self.reused_by_block.get(block_index, 0) + 1
            return x + entry.delta

        result = forward_fn()
        if not hasattr(result, "shape") or tuple(result.shape) != tuple(x.shape):
            raise RuntimeError("Astra block output shape differs from input; cannot cache safely")
        self.recomputed += 1
        self.recomputed_by_block[block_index] = self.recomputed_by_block.get(block_index, 0) + 1
        # When recomputing a block, refresh its entry even during extra warmup.
        self.entries[key] = Entry(
            delta=(result - x).detach(),
            full_step=step, shape=tuple(x.shape), dtype=x.dtype, device=x.device)
        return result

    def stats(self) -> dict[str, Any]:
        total = self.reused + self.recomputed
        return {
            "method": "scheduled", "enabled": self.enabled,
            "selected_blocks": sorted(self.indices),
            "default_period": self.default_period,
            "period_by_block": self.period_by_block,
            "warmup_steps": self.warmup_steps,
            "warmup_by_chunk": self.warmup_by_chunk,
            "max_age": self.max_age,
            "reused_block_calls": self.reused,
            "full_block_calls": self.recomputed,
            "reuse_ratio_selected_calls": self.reused / total if total else 0.0,
            "reused_by_block": self.reused_by_block,
            "recomputed_by_block": self.recomputed_by_block,
            "cross_chunk_reuse": False,
            "warning": "Approximate full-block residual reuse; profile and evaluate long AR quality.",
        }
