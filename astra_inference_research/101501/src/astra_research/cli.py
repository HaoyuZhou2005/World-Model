"""CLI entrypoint; original Astra weights and inference implementation are external."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import uuid

from .config import fingerprint, generation_kwargs, load_config, preflight, resolve_path
from .report import compare, paired_video_metrics
from .identity import source_hashes


def git_revision(repo: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10)
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "UNKNOWN_GIT_REVISION"


def dump_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")


def verified_identity(paths):
    """Require a precomputed manifest, validate source hashes and all weight file sizes.

    Large weights are hashed once by identity.py, not repeatedly during inference.
    This cannot detect same-size file edits AFTER manifest creation; rehash on change.
    """
    doc = json.loads(paths["identity_manifest"].read_text(encoding="utf-8"))
    upstream = source_hashes(paths["astra_repo"])
    if doc.get("astra_source_files_sha256") != upstream:
        raise RuntimeError("Astra source changed since identity manifest generation")
    expected = {
        "astra_ckpt": paths["astra_checkpoint"],
        "wan_dit": paths["wan_model_dir"] / "diffusion_pytorch_model.safetensors",
        "wan_t5": paths["wan_model_dir"] / "models_t5_umt5-xxl-enc-bf16.pth",
        "wan_vae": paths["wan_model_dir"] / "Wan2.1_VAE.pth",
    }
    hashes = {}
    for name, file in expected.items():
        entry = doc.get("weights", {}).get(name)
        if not entry or entry.get("bytes") != file.stat().st_size:
            raise RuntimeError(f"Weight manifest missing or wrong size for {name}")
        digest = entry.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError(f"Weight hash must be a full SHA-256 for {name}")
        hashes[name] = digest
    return upstream, hashes


def run(args: argparse.Namespace) -> Path:
    config = load_config(args.config)
    if args.dry_run:
        print(json.dumps({
            "dry_run": True, "mode": args.mode,
            "paths": config["paths"], "generation": config["generation"],
            "expected_dit": "EternalEvan/Astra:diffsynth/models/wan_video_dit_moe.py",
            "note": "No paths or GPU have been validated; no inference was run."
        }, ensure_ascii=False, indent=2))
        return Path(".")
    paths = preflight(config)
    source_digest, weight_digest = verified_identity(paths)
    if args.mode == "cache" and config["cache"].get("method") != "scheduled":
        raise ValueError("101501 cache mode requires cache.method=scheduled")
    if args.mode == "baseline" and config["cache"].get("method") not in ("none", "residual", "scheduled"):
        raise ValueError("Unexpected cache method")
    output_root = resolve_path(config["paths"].get("output_root", "../runs"), config)
    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / (f"{run_name}_{args.mode}_{uuid.uuid4().hex[:8]}")
    run_dir.mkdir(parents=True, exist_ok=False)
    output_file = run_dir / "generated.mp4"
    kwargs = generation_kwargs(config, paths, output_file)
    sanitized = {key: value for key, value in config.items() if not key.startswith("_")}
    config_hash = fingerprint(config)
    metadata = {
        "status": "started", "mode": args.mode,
        "seed": int(config.get("seed", 42)),
        "config_fingerprint": config_hash,
        "effective_config": sanitized, "generation_kwargs": kwargs,
        "astra_revision": git_revision(paths["astra_repo"]),
        "upstream_files_sha256": source_digest,
        "weights_sha256": weight_digest,
        "torch_profiler": bool(args.profiler),
        "generated_video": str(output_file),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    dump_json(run_dir / "metadata.json", metadata)

    import torch
    from .adapter import run_official, seed_everything
    seed_everything(metadata["seed"])
    start = time.perf_counter()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Astra inference requires a CUDA GPU")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        trace_path = run_dir / "torch_trace.json" if args.profiler else None
        instrument = run_official(
            paths["astra_repo"], kwargs, config,
            cache_enabled=args.mode == "cache",
            trace_path=trace_path,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        metadata.update(
            status="completed", elapsed_seconds=elapsed,
            peak_cuda_memory_bytes=int(torch.cuda.max_memory_allocated()),
            instrumentation=instrument,
            completed_utc=datetime.now(timezone.utc).isoformat(),
        )
        if not output_file.is_file():
            raise FileNotFoundError(f"Official Astra did not create output file: {output_file}")
    except BaseException as error:
        metadata.update(
            status="failed",
            elapsed_seconds=time.perf_counter() - start,
            exception=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        dump_json(run_dir / "metadata.json", metadata)
        print(f"Run metadata: {run_dir / 'metadata.json'}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Astra training-free inference research")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Check configured paths and Astra files")
    doctor.add_argument("--config", required=True)
    runner = sub.add_parser("run", help="Run official Astra with hooks")
    runner.add_argument("--config", required=True)
    runner.add_argument("--mode", choices=("baseline", "cache"), required=True)
    runner.add_argument("--dry-run", action="store_true")
    runner.add_argument("--profiler", action="store_true",
                        help="Expensive torch.profiler trace; do NOT use for speedup measurements")
    comp = sub.add_parser("compare", help="Compare recorded runs")
    comp.add_argument("--reference", required=True)
    comp.add_argument("--candidate", required=True)
    comp.add_argument("--video-metrics", action="store_true")
    args = parser.parse_args()
    if args.command == "doctor":
        cfg = load_config(args.config)
        paths = preflight(cfg)
        source_digest, weight_digest = verified_identity(paths)
        print(json.dumps({"status": "paths_ok", "paths": {k: str(v) for k, v in paths.items()},
                          "astra_git_revision": git_revision(paths["astra_repo"])},
                         indent=2, ensure_ascii=False))
    elif args.command == "run":
        print(f"Output folder: {run(args)}")
    else:
        result = compare(args.reference, args.candidate)
        if args.video_metrics:
            result["pixel_deviation"] = paired_video_metrics(
                Path(args.reference) / "generated.mp4",
                Path(args.candidate) / "generated.mp4")
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
