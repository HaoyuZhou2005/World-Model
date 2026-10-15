"""SHA-256 evidence of checkpoint files and upstream source copies.

Run once per checkpoint, not per inference iteration: hashing large weights
consumes I/O bandwidth on shared research servers.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ASTRA_RELATIVE_FILES = (
    "scripts/infer_demo.py",
    "diffsynth/models/wan_video_dit_moe.py",
    "diffsynth/pipelines/wan_video_recammaster.py",
)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_hashes(astra_repo: str | Path) -> dict[str, str]:
    root = Path(astra_repo)
    result = {}
    for relative in ASTRA_RELATIVE_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required upstream source is missing: {path}")
        result[relative] = sha256_file(path)
    return result


def build_manifest(astra_repo: str | Path, weights: dict[str, str],
                   *, include_weights: bool = True) -> dict[str, Any]:
    paths = {str(k): Path(value).expanduser().resolve() for k, value in weights.items()}
    if not paths:
        raise ValueError("No weights supplied; refusing unpinned experiment")
    manifest = {
        "astra_source_files_sha256": source_hashes(astra_repo),
        "weights": {},
        "note": "Hashes of on-disk artifacts; not a proof of official release provenance.",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Weight not found: {path}")
        manifest["weights"][name] = {
            "filename": path.name, "bytes": path.stat().st_size,
            "sha256": sha256_file(path) if include_weights else None,
        }
    return manifest


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="One-time Astra and Wan weight identity manifest")
    parser.add_argument("--astra-repo", required=True)
    parser.add_argument("--astra-ckpt", required=True)
    parser.add_argument("--wan-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.wan_dir)
    manifest = build_manifest(args.astra_repo, {
        "astra_ckpt": args.astra_ckpt,
        "wan_dit": str(root / "diffusion_pytorch_model.safetensors"),
        "wan_t5": str(root / "models_t5_umt5-xxl-enc-bf16.pth"),
        "wan_vae": str(root / "Wan2.1_VAE.pth"),
    })
    Path(args.out).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                              encoding="utf-8")
    print(f"Wrote identity manifest: {args.out}")


if __name__ == "__main__":
    main()
