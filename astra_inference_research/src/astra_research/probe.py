"""Small global feature fingerprints for cross-chunk *analysis only*.

Global pooled vectors are not aligned tokens. These measurements MUST NOT be
used as evidence that token-level cross-chunk feature reuse is safe.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any


class ChunkProbe:
    def __init__(self, enabled: bool = False, block_indices: list[int] | None = None):
        self.enabled = enabled
        self.blocks = set(block_indices or [])
        self.vectors: dict[tuple[int, int], Any] = {}
        self.records: list[dict[str, Any]] = []
        self.max_records = 256

    def observe(self, block: int, output: Any, *, chunk: int, step: int, branch: int):
        if not self.enabled or block not in self.blocks or step != 0 or branch != 0:
            return
        if len(self.records) >= self.max_records:
            return
        import torch
        with torch.no_grad():
            vector = output.detach().float().mean(dim=tuple(range(output.ndim - 1)))
            vector = vector.cpu()
            previous = self.vectors.get((chunk - 1, block))
            cosine = None
            if previous is not None and previous.shape == vector.shape:
                cosine = float(torch.nn.functional.cosine_similarity(
                    vector.unsqueeze(0), previous.unsqueeze(0)).item())
            self.vectors[(chunk, block)] = vector
            self.records.append({
                "chunk": chunk, "block": block,
                "pooled_mean": float(vector.mean().item()),
                "pooled_std": float(vector.std(unbiased=False).item()),
                "cosine_previous_chunk": cosine,
                "kind": "global pooled summary; tokens NOT aligned",
            })

    def export(self) -> list[dict[str, Any]]:
        return self.records
