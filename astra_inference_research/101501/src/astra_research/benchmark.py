"""Strict offline analysis of completed Astra experiment metadata.

This does not run Astra. We explicitly separate:
- GPU DiT CUDA event ratio (kernel time, not end-to-end)
- end-to-end wall time (includes checkpoint load, VAE and IO)
- pixel difference against an UNCOMPRESSED/identically encoded baseline,
  not FVD, perceptual quality or task/camera fidelity.

For each variant, compare matched seeds/scenes/checkpoints only.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from statistics import median
from typing import Any


def read_completed(path: str | Path) -> dict[str, Any]:
    file = Path(path)
    if file.is_dir():
        file = file / "metadata.json"
    data = json.loads(file.read_text(encoding="utf-8"))
    if data.get("status") != "completed":
        raise ValueError(f"Unfinished experiment: {file}")
    if not isinstance(data.get("instrumentation"), dict):
        raise ValueError(f"No instrumentation record: {file}")
    if float(data.get("elapsed_seconds", 0)) <= 0:
        raise ValueError(f"Invalid elapsed_seconds: {file}")
    data["_source"] = str(file)
    return data


def normalized_config(data: dict[str, Any]) -> dict[str, Any]:
    cfg = deepcopy(data["effective_config"])
    # Only cache/probe may differ; output root is not model input.
    cfg.pop("cache", None)
    cfg.pop("probe", None)
    cfg.get("paths", {}).pop("output_root", None)
    return cfg


def compare_pair(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    mismatches = []
    gpu_a = base.get("gpu_preflight")
    gpu_b = candidate.get("gpu_preflight")
    if gpu_a and gpu_b:
        if (gpu_a.get("index"), gpu_a.get("name")) != (gpu_b.get("index"), gpu_b.get("name")):
            mismatches.append("physical GPU mismatch")
    elif gpu_a or gpu_b:
        mismatches.append("GPU identity only present in one run")
    if base.get("mode") != "baseline" or candidate.get("mode") != "cache":
        mismatches.append("baseline/cache run modes")
    if base.get("seed") != candidate.get("seed"):
        mismatches.append("seed")
    for key in ("astra_revision",):
        left, right = base.get(key), candidate.get(key)
        if not left or left.startswith("UNKNOWN") or not right or right.startswith("UNKNOWN"):
            hashes_a = base.get("upstream_files_sha256")
            hashes_b = candidate.get("upstream_files_sha256")
            if not hashes_a or hashes_a != hashes_b:
                mismatches.append(f"{key} and upstream file hashes not verified")
        elif left != right:
            mismatches.append(key)
    if normalized_config(base) != normalized_config(candidate):
        mismatches.append("generation/input/config mismatch")
    if base.get("torch_profiler") or candidate.get("torch_profiler"):
        mismatches.append("torch.profiler enabled")
    # Optional verified weight manifest digest; both runs must match if present.
    weight_a = base.get("weights_sha256")
    weight_b = candidate.get("weights_sha256")
    if not weight_a or not weight_b:
        mismatches.append("checkpoint SHA256 manifest missing")
    elif weight_a != weight_b:
        mismatches.append("checkpoint SHA256 manifests mismatch")
    b, c = base["instrumentation"], candidate["instrumentation"]
    if b.get("dit_forward_calls") != c.get("dit_forward_calls"):
        mismatches.append("DiT forward count differs")
    bt, ct = b.get("dit_cuda_time_ms_sum"), c.get("dit_cuda_time_ms_sum")
    if bt is not None and (float(bt) <= 0 or ct is None or float(ct) <= 0):
        mismatches.append("invalid DiT CUDA event timing")
    return {
        "baseline": base["_source"], "candidate": candidate["_source"],
        "verified_comparable": not mismatches,
        "mismatches": mismatches,
        "wall_speedup_x": base["elapsed_seconds"] / candidate["elapsed_seconds"],
        "dit_speedup_x": float(bt) / float(ct) if bt and ct else None,
        "baseline_dit_s": float(bt) / 1000 if bt else None,
        "candidate_dit_s": float(ct) / 1000 if ct else None,
        "candidate_cache": c.get("cache"),
        "warning": "Ratios require repeated runs; quality metrics must be independently measured.",
    }


def paired_repeats(bases: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if len(bases) != len(candidates) or len(bases) < 2:
        raise ValueError("At least two matched repeated baseline/cache runs required")
    pairs = [compare_pair(a, b) for a, b in zip(bases, candidates)]
    if any(not p["verified_comparable"] for p in pairs):
        raise ValueError("Cannot report verified speedup: " + repr([p["mismatches"] for p in pairs]))
    wall = [p["wall_speedup_x"] for p in pairs]
    dit = [p["dit_speedup_x"] for p in pairs if p["dit_speedup_x"] is not None]
    return {
        "n_pairs": len(pairs),
        "wall_speedup_median_x": median(wall),
        "dit_speedup_median_x": median(dit) if len(dit) == len(pairs) else None,
        "pairs": pairs,
        "warning": "Small samples: not publication-ready without scene diversity and uncertainty.",
    }
