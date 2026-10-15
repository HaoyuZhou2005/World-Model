"""Per-token cache and true selected-query block execution.

State: per (CFG branch, block), *within a single chunk only*.
Tracks per-token residual and last refresh step; mixed outputs remain ordered
like the original packed [4x|2x|1x|target] sequence.

Not a paper-ready acceleration claim: full K/V projections and full-token
MoE still cost compute; CUDA benchmarking + output quality are mandatory.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import torch

from .gate import TokenGate
from .layout import PackedLayout
from .sparse_block import selected_block_rows


@dataclass
class CacheState:
    delta: torch.Tensor
    prior_input: torch.Tensor
    last_refresh: torch.Tensor
    signature: tuple
    output_dtype: torch.dtype


class TokenResidualCache:
    def __init__(self, gate: TokenGate, selected_blocks: list[int],
                 *, enabled: bool):
        self.gate = gate
        self.selected_blocks = set(selected_blocks)
        self.enabled = enabled
        self.current_chunk: int | None = None
        self.states: dict[tuple[int, int], CacheState] = {}
        self.computed_query_rows = 0
        self.total_query_rows = 0
        self.full_calls = 0
        self.partial_calls = 0
        self.stale_rows = 0
        self.by_stream: dict[str, dict[str, int]] = {}

    def begin_chunk(self, chunk: int):
        if self.current_chunk != chunk:
            self.current_chunk = chunk
            self.states.clear()

    def run(self, block: Any, block_index: int, x: torch.Tensor,
            original_forward: Callable[..., torch.Tensor],
            call_args: tuple, call_kwargs: dict,
            *, layout: PackedLayout, chunk: int, step: int,
            branch: int) -> torch.Tensor:
        if not self.enabled or block_index not in self.selected_blocks:
            return original_forward(x, *call_args, **call_kwargs)
        if x.shape[0] != 1 or x.shape[1] != layout.total:
            raise RuntimeError("Token cache supports one sample and an exact layout")
        self.begin_chunk(chunk)
        key = (branch, block_index)
        state = self.states.get(key)
        if state is not None and (
            state.signature != layout.signature() or
            state.delta.shape != x.shape or state.delta.device != x.device):
            state = None
            self.states.pop(key, None)
        if state is None:
            mask = torch.ones(layout.total, dtype=torch.bool, device=x.device)
        else:
            mask = self.gate.select(
                x, state.prior_input, state.last_refresh,
                layout, chunk=chunk, step=step, previous_delta=state.delta)
        idx = torch.nonzero(mask, as_tuple=False).flatten()
        m, n = int(idx.numel()), layout.total
        self.total_query_rows += n
        self.computed_query_rows += m
        self.stale_rows += n-m
        if m == n or state is None:
            # Crucially, no approximation on refresh/full/first-step.
            result = original_forward(x, *call_args, **call_kwargs)
            if not isinstance(result, torch.Tensor) or result.shape != x.shape:
                raise RuntimeError("Expected a full Astra block output tensor")
            self.full_calls += 1
            self.states[key] = CacheState(
                delta=(result-x).detach().clone(),
                prior_input=x.detach().clone(),
                last_refresh=torch.full((n,),step,device=x.device,dtype=torch.long),
                signature=layout.signature(), output_dtype=result.dtype)
        else:
            # Upstream forward args order:
            # context, cam_emb, t_mod, freqs, modality_inputs,
            # router_weights, router_indices
            def argument(i, name, default=None):
                if name in call_kwargs:
                    return call_kwargs[name]
                return call_args[i] if len(call_args) > i else default
            fresh = selected_block_rows(
                block, x, idx, argument(0,"context"),
                argument(1,"cam_emb"), argument(2,"t_mod"),
                argument(3,"freqs"), argument(4,"modality_inputs"),
                argument(5,"router_weights"), argument(6,"router_indices"))
            # All unselected tokens reuse PREVIOUS denoising residual.
            result = x.to(state.output_dtype) + state.delta.to(state.output_dtype)
            # Fresh selected rows must represent complete block(x) for those rows.
            result[:,idx] = fresh.to(result.dtype)
            state.delta[:,idx] = (fresh - x[:,idx]).detach().to(state.delta.dtype)
            state.last_refresh[idx] = step
            state.prior_input = x.detach().clone()
            self.partial_calls += 1
        for stream in layout.streams:
            counts = self.by_stream.setdefault(
                stream.name, {"fresh_rows":None, "stale_rows":None})
            # Keep counts as CUDA scalar tensors; .item() in every block would
            # force repeated CUDA synchronization and distort cache timings.
            selected = mask[stream.start:stream.stop].sum(dtype=torch.long)
            skipped = torch.tensor(stream.count,device=x.device,dtype=torch.long)-selected
            counts["fresh_rows"] = (selected if counts["fresh_rows"] is None else
                                     counts["fresh_rows"]+selected)
            counts["stale_rows"] = (skipped if counts["stale_rows"] is None else
                                     counts["stale_rows"]+skipped)
        return result

    def statistics(self):
        return {
            "enabled": self.enabled,
            "selected_blocks": sorted(self.selected_blocks),
            "full_block_calls": self.full_calls,
            "partial_block_calls": self.partial_calls,
            "total_query_rows": self.total_query_rows,
            "computed_query_rows": self.computed_query_rows,
            "reused_query_rows": self.stale_rows,
            "query_row_skip_ratio": self.stale_rows / self.total_query_rows
                if self.total_query_rows else 0.0,
            "per_stream": {
                name: {key: int(value.item()) if value is not None else 0
                       for key,value in counts.items()}
                for name,counts in self.by_stream.items()
            },
            "scope": "within denoising trajectory of one chunk only",
            "warning": "Skipped query rows != full DiT FLOPs or end-to-end speedup.",
        }
