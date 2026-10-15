"""Shared GPU safety preflight for the user's A100 research server.

Never auto-select a GPU, never kill/stop/signal other users' processes.
Reject multi-GPU visibility and externally occupied cards before loading Astra.
These checks are snapshot-only; users must still monitor their shared cluster.
"""
from __future__ import annotations

import os
import subprocess
from typing import Any


def selected_physical_gpu(env: str | None) -> int:
    value = (env or "").strip()
    if not value.isdigit():
        raise RuntimeError(
            "Explicitly choose exactly ONE idle physical GPU: "
            "export CUDA_VISIBLE_DEVICES=5  (example only; inspect nvidia-smi first). "
            f"Got CUDA_VISIBLE_DEVICES={value!r}")
    return int(value)


def gpu_snapshot() -> list[dict[str, Any]]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=15,
    )
    devices = []
    for line in result.stdout.splitlines():
        cells = [part.strip() for part in line.split(",")]
        if len(cells) != 5:
            raise RuntimeError(f"Unexpected nvidia-smi output: {line!r}")
        devices.append({
            "index": int(cells[0]), "name": cells[1],
            "memory_used_mib": float(cells[2]),
            "memory_free_mib": float(cells[3]),
            "utilization_pct": float(cells[4]),
        })
    if not devices:
        raise RuntimeError("No GPUs reported by nvidia-smi")
    return devices


def ensure_idle_single_cuda(*, min_free_mib: int = 22000,
                            max_used_mib: int = 256,
                            max_util_pct: int = 10) -> dict[str, Any]:
    physical = selected_physical_gpu(os.environ.get("CUDA_VISIBLE_DEVICES"))
    devices = gpu_snapshot()
    selected = next((x for x in devices if x["index"] == physical), None)
    if selected is None:
        raise RuntimeError(f"GPU {physical} not found; visible physical cards={devices}")
    if selected["memory_used_mib"] > max_used_mib or (
        selected["utilization_pct"] > max_util_pct
    ) or selected["memory_free_mib"] < min_free_mib:
        raise RuntimeError(
            f"Refusing to use busy/low-VRAM GPU {physical}: {selected}. "
            "Choose an idle GPU manually; do not kill others' processes.")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "CUDA must see exactly one GPU after masking. "
            f"torch.cuda.device_count()={torch.cuda.device_count()}")
    return selected
