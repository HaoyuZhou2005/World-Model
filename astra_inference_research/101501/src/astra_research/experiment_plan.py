"""Generate deterministic and isolated cache ablation configs.

Never silently guess checkpoints or input paths. The generated configs inherit
the template, so use a PRIVATE valid template on the server and keep it untracked.
Only planned configs are written; this module NEVER occupies a GPU.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

DEFAULT_VARIANTS: dict[str, dict] = {
    "baseline": {"mode": "baseline", "cache": {"method": "scheduled",
        "block_indices": list(range(30)), "default_period": 2, "warmup_steps": 1, "max_age": 1}},
    "uniform_i2_k1": {"mode": "cache", "cache": {"method": "scheduled",
        "block_indices": list(range(30)), "default_period": 2, "warmup_steps": 1, "max_age": 1}},
    "uniform_i2_k2": {"mode": "cache", "cache": {"method": "scheduled",
        "block_indices": list(range(30)), "default_period": 2, "warmup_steps": 2, "max_age": 1}},
    "uniform_i2_k4": {"mode": "cache", "cache": {"method": "scheduled",
        "block_indices": list(range(30)), "default_period": 2, "warmup_steps": 4, "max_age": 1}},
    "protect_first_last_k4": {"mode": "cache", "cache": {"method": "scheduled",
        "block_indices": list(range(30)), "default_period": 2, "warmup_steps": 4,
        "max_age": 1, "period_by_block": {"0": 1, "29": 1}}},
    "early_full_last_half_i2": {"mode": "cache", "cache": {"method": "scheduled",
        "block_indices": list(range(15, 30)), "default_period": 2,
        "warmup_steps": 4, "max_age": 1}},
    "late_full_first_half_i2": {"mode": "cache", "cache": {"method": "scheduled",
        "block_indices": list(range(0, 15)), "default_period": 2,
        "warmup_steps": 4, "max_age": 1}},
    "progressive_k": {"mode": "cache", "cache": {"method": "scheduled",
        "block_indices": list(range(30)), "default_period": 2,
        "warmup_steps": 1, "warmup_by_chunk": {"0": 1, "1": 2, "2": 4},
        "max_age": 1}},
}


def generate(template: dict, output_root: Path, *, num_blocks: int,
             names: list[str] | None = None) -> dict:
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {}
    names = names or list(DEFAULT_VARIANTS)
    for name in names:
        if name not in DEFAULT_VARIANTS:
            raise ValueError(f"Unknown variant {name}")
        spec = deepcopy(DEFAULT_VARIANTS[name])
        config = deepcopy(template)
        c = spec["cache"]
        # 30 is the verified block count in garden_1 experiment, but it is NOT
        # a universal Astra architecture property. Explicit --num-blocks is required.
        if c["block_indices"] == list(range(30)):
            c["block_indices"] = list(range(num_blocks))
        else:
            c["block_indices"] = [i for i in c["block_indices"] if i < num_blocks]
        if name == "protect_first_last_k4":
            c["period_by_block"] = {"0": 1, str(num_blocks - 1): 1}
        if "period_by_block" in c:
            c["period_by_block"] = {
                key: value for key, value in c["period_by_block"].items()
                if int(key) < num_blocks
            }
        config["cache"] = c
        config["probe"] = {"enabled": False, "block_indices": []}
        path = output_root / f"{name}.json"
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest[name] = {"mode": spec["mode"], "config": str(path)}
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-blocks", required=True, type=int,
                        help="Read len(pipe.dit.blocks) from verified running checkpoint")
    parser.add_argument("--variants", nargs="*")
    args = parser.parse_args()
    cfg = json.loads(Path(args.template).read_text(encoding="utf-8"))
    manifest = generate(cfg, Path(args.out), num_blocks=args.num_blocks, names=args.variants)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
