"""Streaming per-chunk deviation diagnostics, never ground-truth model quality.

One-pass decoder avoids loading long RGB videos in RAM.
No MP4 included in uploaded 26101501.zip: use server video files.
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


def evaluate_pair(reference: str | Path, candidate: str | Path, *,
                  rgb_frames_per_chunk: int, n_chunks: int,
                  condition_frames: int = 1) -> dict[str, Any]:
    if rgb_frames_per_chunk < 1 or n_chunks < 1 or condition_frames < 0:
        raise ValueError("Invalid chunk or condition frame counts")
    import numpy as np
    from itertools import zip_longest

    expected = condition_frames + n_chunks * rgb_frames_per_chunk
    sentinel = object()
    seen = 0
    shape = None
    sq = [0.0] * n_chunks
    absolute = [0.0] * n_chunks
    pixel_count = [0] * n_chunks
    temporal_ref = [0.0] * n_chunks
    temporal_cache = [0.0] * n_chunks
    temporal_n = [0] * n_chunks
    previous = None

    # Ref and cached videos must have EXACTLY the same number of frames.
    # No silent truncation at the shorter video.
    for fa, fb in zip_longest(_frames(Path(reference)), _frames(Path(candidate)),
                             fillvalue=sentinel):
        if fa is sentinel or fb is sentinel:
            raise ValueError("Reference and candidate frame counts differ")
        if fa.shape != fb.shape:
            raise ValueError(f"Frame shape mismatch: {fa.shape} vs {fb.shape}")
        if shape is None:
            shape = tuple(fa.shape)
            if len(shape) != 3 or shape[-1] != 3:
                raise ValueError(f"Expected RGB HWC frame, got {shape}")
        if tuple(fa.shape) != shape:
            raise ValueError("Frame shape changed within video")
        if seen >= expected:
            raise ValueError(f"More frames than expected={expected}; check VAE frame count")
        if seen >= condition_frames:
            index = (seen - condition_frames) // rgb_frames_per_chunk
            diff = fa.astype(np.float64) - fb.astype(np.float64)
            sq[index] += float(np.square(diff).sum())
            absolute[index] += float(np.abs(diff).sum())
            pixel_count[index] += int(diff.size)
            # Frame-to-frame proxy entirely WITHIN the current generated chunk.
            if previous is not None and (seen - condition_frames) % rgb_frames_per_chunk != 0:
                temporal_ref[index] += float(np.abs(fa - previous[0]).mean())
                temporal_cache[index] += float(np.abs(fb - previous[1]).mean())
                temporal_n[index] += 1
        previous = fa, fb
        seen += 1
    if seen != expected:
        raise ValueError(f"Expected exactly {expected} RGB frames, found {seen}")
    chunks = []
    for k in range(n_chunks):
        mse = sq[k] / pixel_count[k]
        chunks.append({
            "chunk": k,
            "start": condition_frames + k * rgb_frames_per_chunk,
            "stop": condition_frames + (k + 1) * rgb_frames_per_chunk,
            "mae_to_no_cache": absolute[k] / pixel_count[k],
            "rmse_to_no_cache": mse ** 0.5,
            "psnr_to_no_cache_db": None if mse == 0 else 10 * log10(255.0 ** 2 / mse),
            "perfect_match": mse == 0,
            "flicker_proxy_reference": temporal_ref[k] / temporal_n[k] if temporal_n[k] else None,
            "flicker_proxy_candidate": temporal_cache[k] / temporal_n[k] if temporal_n[k] else None,
        })
    return {
        "reference": str(reference), "candidate": str(candidate),
        "frames": seen, "condition_frames": condition_frames,
        "rgb_frames_per_chunk": rgb_frames_per_chunk,
        "n_chunks": n_chunks, "per_chunk": chunks,
        "mean_chunk_mae": sum(x["mae_to_no_cache"] for x in chunks) / n_chunks,
        "important": (
            "All deviations are versus a no-cache GENERATED video, NOT ground-truth "
            "accuracy or FVD/VBench. Lower flicker proxy can mean blur."
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
    report = evaluate_pair(args.reference, args.candidate,
                           rgb_frames_per_chunk=args.rgb_per_chunk,
                           n_chunks=args.chunks,
                           condition_frames=args.condition_frames)
    out = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.out:
        Path(args.out).write_text(out, encoding="utf-8")
    else:
        print(out)


if __name__ == "__main__":
    main()
