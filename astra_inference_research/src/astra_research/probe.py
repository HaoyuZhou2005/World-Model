"""Optional pooled activation similarity diagnostics.

Both intra-step and inter-chunk summaries are descriptive (NOT aligned tokens).
Collecting them requires GPU->CPU synchronization; DISABLE for speed benchmarks.
"""
from __future__ import annotations
from typing import Any


class ChunkProbe:
    def __init__(self, enabled: bool = False, block_indices: list[int] | None = None,
                 step_analysis: bool = False, max_records: int = 5000):
        self.enabled = enabled
        self.blocks = set(block_indices or [])
        self.step_analysis = step_analysis
        self.previous_step: dict[tuple[int, int, int], Any] = {}
        self.previous_chunk: dict[tuple[int, int], Any] = {}
        self.records: list[dict[str, Any]] = []
        self.max_records = max_records

    def observe(self, block: int, output: Any, *, chunk: int, step: int, branch: int):
        if not self.enabled or block not in self.blocks:
            return
        if not self.step_analysis and (step != 0 or branch != 0):
            return
        if len(self.records) >= self.max_records:
            return
        import torch
        with torch.no_grad():
            vector = output.detach().float().mean(dim=tuple(range(output.ndim - 1)))
            vector = vector.cpu()
            record: dict[str, Any] = {
                "chunk": chunk, "step": step, "branch": branch, "block": block,
                "pooled_mean": float(vector.mean().item()),
                "pooled_std": float(vector.std(unbiased=False).item()),
                "kind": "global pooled activation summary; token positions NOT aligned",
            }
            if self.step_analysis:
                key = (chunk, branch, block)
                previous = self.previous_step.get(key)
                if previous is not None and previous.shape == vector.shape:
                    record["cosine_previous_step"] = float(
                        torch.nn.functional.cosine_similarity(
                            vector.unsqueeze(0), previous.unsqueeze(0)).item())
                    record["l1_previous_step"] = float((vector - previous).abs().mean().item())
                self.previous_step[key] = vector
            if step == 0 and branch == 0:
                previous = self.previous_chunk.get((chunk - 1, block))
                if previous is not None and previous.shape == vector.shape:
                    record["cosine_previous_chunk"] = float(
                        torch.nn.functional.cosine_similarity(
                            vector.unsqueeze(0), previous.unsqueeze(0)).item())
                self.previous_chunk[(chunk, block)] = vector
            self.records.append(record)

    def export(self) -> list[dict[str, Any]]:
        return self.records
