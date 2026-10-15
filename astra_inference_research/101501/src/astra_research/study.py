"""Offline study commands: generate configs, compare matched replicates, video fidelity.

No GPUs are touched by the offline commands.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import read_completed, paired_repeats


def main() -> None:
    parser = argparse.ArgumentParser(description="101501 offline experiments")
    sub = parser.add_subparsers(dest="task", required=True)
    matrix = sub.add_parser("configs")
    matrix.add_argument("--template", required=True)
    matrix.add_argument("--out", required=True)
    matrix.add_argument("--num-blocks", type=int, required=True)
    evaluate = sub.add_parser("videos")
    evaluate.add_argument("--reference", required=True)
    evaluate.add_argument("--candidate", required=True)
    evaluate.add_argument("--rgb-per-chunk", type=int, required=True)
    evaluate.add_argument("--chunks", type=int, required=True)
    evaluate.add_argument("--condition-frames", type=int, default=1)
    paired = sub.add_parser("pairs")
    paired.add_argument("--baseline", nargs="+", required=True,
                        help="Ordered directories or metadata.json paths")
    paired.add_argument("--cache", nargs="+", required=True)
    args = parser.parse_args()
    if args.task == "configs":
        from .experiment_plan import generate
        template = json.loads(Path(args.template).read_text(encoding="utf-8"))
        result = generate(template, Path(args.out), num_blocks=args.num_blocks,
                          template_base=Path(args.template).expanduser().resolve().parent)
    elif args.task == "videos":
        from .evaluate import evaluate_pair
        result = evaluate_pair(args.reference, args.candidate,
                               rgb_frames_per_chunk=args.rgb_per_chunk,
                               n_chunks=args.chunks,
                               condition_frames=args.condition_frames)
    else:
        result = paired_repeats([read_completed(x) for x in args.baseline],
                                [read_completed(x) for x in args.cache])
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
