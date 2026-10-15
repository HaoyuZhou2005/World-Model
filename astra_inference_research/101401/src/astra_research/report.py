"""Compare two recorded inference runs. Not a ground-truth world-model evaluator."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_run(folder: str | Path) -> dict[str, Any]:
    p = Path(folder) / "metadata.json"
    with p.open(encoding="utf-8") as file:
        result = json.load(file)
    if result.get("status") != "completed":
        raise ValueError(f"Run is not completed: {p}")
    return result


def compare(reference: str | Path, candidate: str | Path) -> dict[str, Any]:
    base = _read_run(reference)
    other = _read_run(candidate)
    keys = ("config_fingerprint", "astra_revision", "seed")
    unequal = [key for key in keys if base.get(key) != other.get(key)]
    # Cache fields necessarily differ: compare normalized configs separately.
    cfg_a, cfg_b = base["effective_config"], other["effective_config"]
    from copy import deepcopy
    cfg_a, cfg_b = deepcopy(cfg_a), deepcopy(cfg_b)
    cfg_a.pop("cache", None)
    cfg_b.pop("cache", None)
    cfg_a.pop("probe", None)
    cfg_b.pop("probe", None)
    if cfg_a != cfg_b:
        unequal.append("effective_config_except_cache_and_probe")
    if base["mode"] != "baseline" or other["mode"] != "cache":
        unequal.append("expected_baseline_then_cache")
    if base.get("torch_profiler") or other.get("torch_profiler"):
        unequal.append("profiler_enabled_biased_timing")
    speedup = float(base["elapsed_seconds"]) / float(other["elapsed_seconds"])
    result = {
        "reference": str(reference),
        "candidate": str(candidate),
        "comparable": not unequal or unequal == ["config_fingerprint"],
        "mismatches": unequal,
        "reference_seconds": base["elapsed_seconds"],
        "candidate_seconds": other["elapsed_seconds"],
        "speedup_x": speedup,
        "reference_dit_calls": base["instrumentation"]["dit_forward_calls"],
        "candidate_dit_calls": other["instrumentation"]["dit_forward_calls"],
        "reference_peak_gpu_gb": base.get("peak_cuda_memory_bytes", 0) / (1024**3),
        "candidate_peak_gpu_gb": other.get("peak_cuda_memory_bytes", 0) / (1024**3),
        "cache": other["instrumentation"]["cache"],
        "disclaimer": (
            "Speedup is an observed single-run wall-time ratio, not statistically reliable. "
            "Repeat warm runs; use matching seeds/inputs/hardware and quality tests. "
            "Cache output differs numerically by design."
        ),
    }
    return result


def paired_video_metrics(reference_mp4: str | Path, candidate_mp4: str | Path,
                         max_frames: int | None = None) -> dict[str, Any]:
    """Pixel deviation from no-cache output, NOT PSNR against ground truth.

    Stochastic/autoregressive samples may legitimately differ; interpret carefully.
    """
    import numpy as np
    import imageio.v2 as imageio
    from math import log10

    ref = imageio.get_reader(str(reference_mp4))
    cand = imageio.get_reader(str(candidate_mp4))
    n, squared_sum, absolute_sum, element_count = 0, 0.0, 0.0, 0
    try:
        while max_frames is None or n < max_frames:
            try:
                a = np.asarray(ref.get_next_data(), dtype=np.float32)
                b = np.asarray(cand.get_next_data(), dtype=np.float32)
            except IndexError:
                break
            except StopIteration:
                break
            if a.shape != b.shape:
                raise ValueError(f"Frame shape differs: {a.shape} vs {b.shape}")
            diff = a - b
            squared_sum += float(np.square(diff, dtype=np.float64).sum())
            absolute_sum += float(np.abs(diff).sum())
            element_count += diff.size
            n += 1
        if n == 0:
            raise ValueError("No comparable video frames")
        mse = squared_sum / element_count
        return {
            "frames_compared": n,
            "mse_to_reference_video": mse,
            "mae_to_reference_video": absolute_sum / element_count,
            "psnr_to_reference_video_db": float("inf") if mse == 0 else 10 * log10(255.0**2 / mse),
            "warning": "No-cache reference comparison; NOT a ground-truth generative quality metric.",
        }
    finally:
        ref.close()
        cand.close()
