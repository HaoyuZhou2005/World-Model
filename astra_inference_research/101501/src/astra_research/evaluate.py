"""Strict matched-reference video diagnostics for autoregressive rollouts.

No-cache generated video is the REFERENCE, not ground-truth future.
PSNR/MAE here measure deviation caused by caching, not absolute realism.
Flicker proxy can improve when images get over-smoothed; never label it "quality".
"""
from __future__ import annotations

import json
from math import log10
from pathlib import Path
from typing import Any


def _frames(path: Path):
    import imageio.v2 as imageio
    import numpy as np

    reader = imageio.get_reader(str(path))
    try:
        while True:
            try:
                frame = reader.get_next_data()
            except (IndexError, StopIteration):
                return
            yield np.asarray(frame, dtype=np.float32)
    finally:
        reader.close()


def evaluate_pair(
    reference: str | Path, candidate: str | Path, *,
    rgb_frames_per_chunk: int, n_chunks: int,
    condition_frames: int = 1,
) -> dict[str, Any]:
    if rgb_frames_per_chunk < 1 or n_chunks < 1 or condition_frames < 0:
        raise ValueError("Invalid chunk or condition frame counts")
    import numpy as np

    a = list(_frames(Path(reference)))
    b = list(_frames(Path(candidate)))
    expected = n_chunks * rgb_frames_per_chunk + condition_frames
    if len(a) != expected or len(b) != expected:
        raise ValueError(
            f"Expected exactly {expected} RGB frames: "
            f"reference={len(a)} candidate={len(b)}. "
            "Check VAE temporal compression and official video frame accounting.")
    shape = a[0].shape
    if any(frame.shape != shape for frame in a + b):
        raise ValueError("Different frame shapes: do not silently resize for research results")
    if len(shape) != 3 or shape[-1] != 3:
        raise ValueError(f"Expected RGB HWC frame, got {shape}")
    per_chunk = []
    ref_flicker = []
    cached_flicker = []
    for chunk in range(n_chunks):
        start = condition_frames + chunk * rgb_frames_per_chunk
        stop = start + rgb_frames_per_chunk
        squared = 0.0
        absolute = 0.0
        count = 0
        for i in range(start, stop):
            diff = a[i].astype(np.float64) - b[i].astype(np.float64)
            squared += float(np.square(diff).sum())
            absolute += float(np.abs(diff).sum())
            count += int(diff.size)
        mse = squared / count
        per_chunk.append({
            "chunk": chunk, "start": start, "stop": stop,
            "mae_to_no_cache": absolute / count,
            "rmse_to_no_cache": mse ** 0.5,
            "psnr_to_no_cache_db": None if mse == 0 else 10 * log10(255 ** 2 / mse),
            "perfect_match": mse == 0,
        })
        # Frame-difference over the same contiguous generated range only;
        # not a ground-truth temporal coherence metric.
        if stop - start >= 2:
            ref_flicker.append(sum(float(np.abs(a[i] - a[i-1]).mean())
                                   for i in range(start + 1, stop)) / (stop - start - 1))
            cached_flicker.append(sum(float(np.abs(b[i] - b[i-1]).mean())
                                      for i in range(start + 1, stop)) / (stop - start - 1))
        else:
            ref_flicker.append(None)
            cached_flicker.append(None)
    return {
        "reference": str(reference), "candidate": str(candidate),
        "frames": len(a), "condition_frames": condition_frames,
        "rgb_frames_per_chunk": rgb_frames_per_chunk, "n_chunks": n_chunks,
        "per_chunk": per_chunk,
        "mean_chunk_mae": sum(x["mae_to_no_cache"] for x in per_chunk) / n_chunks,
        "flicker_proxy_by_chunk_reference": ref_flicker,
        "flicker_proxy_by_chunk_candidate": cached_flicker,
        "important": (
            "All fidelity values are relative to the no-cache video; "
            "not FVD, VBench, action following or camera-pose accuracy. "
            "Reduced frame difference can indicate blur, not improvement."
        ),
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--rgb-per-chunk", type=int, required=True)
    parser.add_argument("--chunks", type=int, required=True)
    parser.add_argument("--condition-frames", type=int, default=1)
    parser.add_argument("--out")
    args = parser.parse_args()
    result = evaluate_pair(
        args.reference, args.candidate,
        rgb_frames_per_chunk=args.rgb_per_chunk,
        n_chunks=args.chunks, condition_frames=args.condition_frames)
    content = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.out:
        Path(args.out).write_text(content, encoding="utf-8")
    else:
        print(content)


if __name__ == "__main__":
    main()
