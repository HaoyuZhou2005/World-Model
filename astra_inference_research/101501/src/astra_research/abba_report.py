"""Turn an ABBA experiment's output manifest into matched latency statistics.

Pair A1 with B1 and A2 with B2 per cycle; never compare a cold-start
single run to a warm-cache single run as if they were repeated trials.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import paired_repeats, read_completed


def summarize_abba(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("dry_run"):
        raise ValueError("ABBA manifest is dry-run only; no GPU inference evidence")
    runs = manifest.get("runs", [])
    if not runs or len(runs) % 4 != 0:
        raise ValueError("ABBA runs must be complete 4-run cycles")
    baselines, candidates = [], []
    for pos in range(0, len(runs), 4):
        group = runs[pos:pos+4]
        if [r.get("mode") for r in group] != [
            "baseline", "cache", "cache", "baseline"
        ]:
            raise ValueError(f"Bad ABBA order in cycle {pos//4}")
        if any(r.get("exit_code") != 0 or not r.get("run_dir") for r in group):
            raise ValueError(f"Incomplete ABBA cycle {pos//4}")
        baselines.extend([read_completed(group[0]["run_dir"]),
                          read_completed(group[3]["run_dir"])])
        candidates.extend([read_completed(group[1]["run_dir"]),
                           read_completed(group[2]["run_dir"])])
    summary = paired_repeats(baselines, candidates)
    summary["abba_cycles"] = len(runs) // 4
    summary["provenance"] = str(path)
    summary["note"] = (
        "Matched ABBA ordering and file identities only. "
        "Still require multiple scenes, real benchmark quality and idle GPU."
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True,
                        help="--logs/abba_results.json produced by abba.py --execute")
    parser.add_argument("--out", help="Optional JSON output path")
    args = parser.parse_args()
    summary = summarize_abba(args.manifest)
    content = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(content, encoding="utf-8")
    else:
        print(content)


if __name__ == "__main__":
    main()
