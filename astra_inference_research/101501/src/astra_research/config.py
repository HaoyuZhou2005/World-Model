"""Configuration and fail-closed validation; never silently guess local paths."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REQUIRED_PATHS = ("astra_repo", "astra_checkpoint", "wan_model_dir", "condition_image")
REQUIRED_WAN_FILES = (
    "diffusion_pytorch_model.safetensors",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.1_VAE.pth",
)
GENERATE_KEYS = {
    "condition_pth", "condition_video", "condition_image", "dit_path",
    "wan_model_path", "output_path", "start_frame",
    "initial_condition_frames", "frames_per_generation",
    "total_frames_to_generate", "max_history_frames", "device",
    "prompt", "modality_type", "use_real_poses", "scene_info_path",
    "use_camera_cfg", "camera_guidance_scale", "text_guidance_scale",
    "moe_num_experts", "moe_top_k", "moe_hidden_dim", "cam_type",
    "use_gt_prompt", "add_icons",
}


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)
    for section in ("paths", "generation", "cache", "measurement"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Missing JSON object: {section}")
    unexpected = set(config["generation"]) - GENERATE_KEYS
    if unexpected:
        raise ValueError(f"Unsupported generation keys: {sorted(unexpected)}")
    if config["generation"].get("total_frames_to_generate", 0) <= 0:
        raise ValueError("total_frames_to_generate must be positive (latent frames)")
    if config["generation"].get("frames_per_generation", 0) <= 0:
        raise ValueError("frames_per_generation must be positive (latent frames)")
    cache = config["cache"]
    if cache.get("method", "none") not in ("none", "residual"):
        raise ValueError("Only none/residual caching is implemented")
    if int(cache.get("interval", 2)) < 2:
        raise ValueError("cache.interval must be >= 2")
    if any(int(x) < 0 for x in cache.get("block_indices", [])):
        raise ValueError("cache.block_indices must be nonnegative")
    config["_config_file"] = str(config_path)
    return config


def resolve_path(value: str, config: dict[str, Any]) -> Path:
    text = str(value)
    if not text or "TODO_" in text or "<" in text or ">" in text:
        raise ValueError(f"Unresolved path placeholder: {text!r}")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = Path(config["_config_file"]).parent / path
    return path.resolve()


def preflight(config: dict[str, Any]) -> dict[str, Path]:
    paths = {}
    for key in REQUIRED_PATHS:
        value = config["paths"].get(key)
        if not value:
            raise ValueError(f"paths.{key} must be provided")
        paths[key] = resolve_path(value, config)
        if not paths[key].exists():
            raise FileNotFoundError(f"paths.{key} does not exist: {paths[key]}")
    astra_script = paths["astra_repo"] / "scripts" / "infer_demo.py"
    model_file = paths["astra_repo"] / "diffsynth" / "models" / "wan_video_dit_moe.py"
    for name, path in (("Astra script", astra_script), ("Astra MoE model", model_file)):
        if not path.is_file():
            raise FileNotFoundError(f"{name} not found: {path}")
    for filename in REQUIRED_WAN_FILES:
        if not (paths["wan_model_dir"] / filename).is_file():
            raise FileNotFoundError(f"Wan model file missing: {filename}")
    if not paths["astra_checkpoint"].is_file():
        raise FileNotFoundError("Astra checkpoint must be a file")
    if not paths["condition_image"].is_file():
        raise FileNotFoundError("Official infer_demo requires an existing --condition_image")
    if config["generation"].get("condition_pth"):
        pth = resolve_path(config["generation"]["condition_pth"], config)
        if not pth.is_file():
            raise FileNotFoundError(str(pth))
        paths["condition_pth"] = pth
    return paths


def generation_kwargs(config: dict[str, Any], paths: dict[str, Path], output_file: Path) -> dict[str, Any]:
    kw = dict(config["generation"])
    kw.update(
        condition_image=str(paths["condition_image"]),
        dit_path=str(paths["astra_checkpoint"]),
        wan_model_path=str(paths["wan_model_dir"]),
        output_path=str(output_file),
    )
    if "condition_pth" in paths:
        kw["condition_pth"] = str(paths["condition_pth"])
    if kw.get("scene_info_path"):
        kw["scene_info_path"] = str(resolve_path(kw["scene_info_path"], config))
    if kw.get("condition_video"):
        kw["condition_video"] = str(resolve_path(kw["condition_video"], config))
    return kw


def fingerprint(config: dict[str, Any]) -> str:
    public = {key: value for key, value in config.items() if not key.startswith("_")}
    return hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()
