"""Fail-closed single-idle-GPU research-server preflight.

No GPU auto-selection, process termination, or scheduling. This is only a
snapshot; a shared server can become busy after the check.
"""
from __future__ import annotations
import os
import subprocess


def one_idle_gpu(*, minimum_free_mib: int=20000, maximum_used_mib: int=512,
                 maximum_util_percent: float=15) -> dict:
    physical=os.environ.get("CUDA_VISIBLE_DEVICES","")
    if not physical.isdecimal():
        raise RuntimeError("Choose exactly one idle physical GPU with CUDA_VISIBLE_DEVICES=N")
    result=subprocess.run(["nvidia-smi","--query-gpu=index,name,memory.used,memory.free,utilization.gpu",
                           "--format=csv,noheader,nounits"],
                          text=True,capture_output=True,check=True,timeout=15)
    candidates=[]
    for line in result.stdout.strip().splitlines():
        fields=[s.strip() for s in line.split(",")]
        if len(fields)!=5:
            raise RuntimeError(f"Unrecognized nvidia-smi query result: {line}")
        candidates.append({
            "index":int(fields[0]),"name":fields[1],
            "used_mib":float(fields[2]),
            "free_mib":float(fields[3]),
            "util_percent":float(fields[4])})
    selected=next((g for g in candidates if g["index"]==int(physical)),None)
    if selected is None:
        raise RuntimeError(f"Selected physical GPU {physical} not found")
    if (selected["used_mib"]>maximum_used_mib or
        selected["free_mib"]<minimum_free_mib or
        selected["util_percent"]>maximum_util_percent):
        raise RuntimeError(f"Refusing occupied or low-memory GPU {physical}: {selected}")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError("Exactly ONE logical CUDA GPU must be available")
    return selected
