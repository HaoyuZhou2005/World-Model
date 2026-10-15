"""Sequential, opt-in ABBA run driver for the shared one-GPU research server.

By default this ONLY prints a plan. --execute is required for expensive
inference. Each subprocess has to pass the CLI's independent idle-GPU check.
Never sends signals, stops jobs, uses another GPU, or trains weights.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


def abba_sequence(baseline_config: str, cache_config: str, cycles: int) -> list[dict[str, Any]]:
    if cycles < 1:
        raise ValueError("cycles must be >= 1")
    sequence = []
    for cycle in range(cycles):
        # ABBA controls some run-order drift, not thermal or multi-scene bias.
        for order, (mode, config) in enumerate((
            ("baseline", baseline_config),
            ("cache", cache_config),
            ("cache", cache_config),
            ("baseline", baseline_config),
        )):
            sequence.append({
                "cycle": cycle, "order": order,
                "mode": mode, "config": str(config),
            })
    return sequence


def validate_manifest(manifest_path: Path, base: str, method: str) -> tuple[str, str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if base not in manifest or method not in manifest:
        raise KeyError(f"Unknown variants: base={base}, cache={method}")
    if manifest[base].get("mode") != "baseline" or manifest[method].get("mode") != "cache":
        raise ValueError("Variants must be a baseline and a cache run")
    return str(manifest[base]["config"]), str(manifest[method]["config"])


def run(sequence: list[dict], *, log_dir: Path, execute: bool) -> dict[str, Any]:
    log_dir = log_dir.resolve()
    summary = {
        "dry_run": not execute,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sequence": sequence,
        "runs": [],
    }
    if not execute:
        return summary
    from .safety import ensure_idle_single_cuda
    # External process cannot grant GPU access: user must explicitly mask one idle GPU.
    gpu = ensure_idle_single_cuda()
    log_dir.mkdir(parents=True, exist_ok=True)
    summary["initial_gpu"] = gpu
    report_file = log_dir / "abba_results.json"

    for number, item in enumerate(sequence):
        cmd = [
            sys.executable, "-m", "astra_research.cli", "run",
            "--config", str(item["config"]), "--mode", item["mode"],
        ]
        # No shell commands from paths: subprocess arguments are passed as a list.
        completed = subprocess.run(cmd, text=True, capture_output=True,
                                   stdin=subprocess.DEVNULL, check=False)
        log_file = log_dir / f"{number:03d}_{item['mode']}.log"
        log_file.write_text(completed.stdout + "\n--- STDERR ---\n" + completed.stderr,
                            encoding="utf-8")
        candidates = re.findall(r"^Output folder:\s*(.+)$", completed.stdout, flags=re.M)
        status = {
            **item, "number": number, "exit_code": completed.returncode,
            "log_file": str(log_file),
            "run_dir": candidates[-1].strip() if candidates else None,
        }
        summary["runs"].append(status)
        report_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                               encoding="utf-8")
        if completed.returncode != 0 or not status["run_dir"]:
            raise RuntimeError(
                f"ABBA stopped on run {number}, exit={completed.returncode}. "
                f"Review log: {log_file}. No additional jobs were launched.")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="101501 sequential ABBA inference driver")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--cache", default="uniform_i2_k4")
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--logs", required=True, help="Prefer free disk, e.g. /data6/tmp/...")
    parser.add_argument("--execute", action="store_true",
                        help="Without this flag the command NEVER starts a GPU job")
    args = parser.parse_args()
    base, candidate = validate_manifest(
        Path(args.manifest), args.baseline, args.cache)
    sequence = abba_sequence(base, candidate, args.cycles)
    result = run(sequence, log_dir=Path(args.logs), execute=args.execute)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
