"""Invoke the ORIGINAL Astra inference routine with non-invasive hooks.

All model parameters, scheduler, FramePack, MoAE, camera actions and output
decoding are from upstream. No training/retraining is performed here.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import uuid
from unittest.mock import patch

from .instrument import TokenInstrumentation


@contextmanager
def import_upstream(repo):
    repo = str(Path(repo).resolve())
    inserted = repo not in sys.path
    if inserted:
        sys.path.insert(0, repo)
    try:
        yield
    finally:
        if inserted:
            sys.path.remove(repo)


def resolve_paths(config):
    results = {}
    for key in ("astra_repo", "astra_checkpoint", "wan_model_dir", "condition_image"):
        text = str(config["paths"].get(key,""))
        if not text or "TODO_" in text or "<" in text:
            raise ValueError(f"Unresolved local path: paths.{key}={text}")
        path = Path(text).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"paths.{key}: {path}")
        results[key] = path
    for relative in ("scripts/infer_demo.py","diffsynth/models/wan_video_dit_moe.py"):
        if not (results["astra_repo"]/relative).is_file():
            raise FileNotFoundError(f"Official Astra source missing: {relative}")
    for name in ("diffusion_pytorch_model.safetensors",
                 "models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth"):
        if not (results["wan_model_dir"]/name).is_file():
            raise FileNotFoundError(f"Missing Wan component {name}")
    if not results["astra_checkpoint"].is_file() or not results["condition_image"].is_file():
        raise FileNotFoundError("Checkpoint/condition image must be existing files")
    return results


def validate_settings(config):
    for name in ("paths","generation","token_cache","measurement"):
        if not isinstance(config.get(name),dict):
            raise ValueError(f"Missing section {name}")
    g = config["generation"]
    for key in ("frames_per_generation","total_frames_to_generate"):
        if int(g.get(key,0)) < 1:
            raise ValueError(f"{key} must be >=1")
    if int(config["measurement"].get("steps",50)) != 50:
        raise ValueError("Astra upstream script hardcodes 50 denoising steps")
    if not isinstance(config["token_cache"].get("ratios"), dict):
        raise ValueError("token_cache.ratios required")
    from .gate import GateConfig
    cache = config["token_cache"]
    GateConfig(ratios=cache["ratios"],
               min_per_stream=int(cache.get("min_per_stream",1)),
               first_full_steps=int(cache.get("first_full_steps",1)),
               max_age=int(cache.get("max_age",1)),
               coarse_threshold=cache.get("coarse_threshold"),
               horizon_boost=float(cache.get("horizon_boost",0.0))).validate()
    if config["token_cache"].get("policy","change_topk") != "change_topk":
        raise ValueError("Only change_topk heuristic executable; untrained MLP gate forbidden")
    if float(g.get("text_guidance_scale",1.0)) != 1.0:
        raise ValueError("Text CFG>1 unsupported")
    if "TODO_" in str(g.get("prompt","")):
        raise ValueError("Fill in benchmark prompt to avoid accidental mismatched conditions")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for buf in iter(lambda: f.read(4*1024*1024),b""):
            h.update(buf)
    return h.hexdigest()


def source_provenance(paths):
    names = ("scripts/infer_demo.py","diffsynth/models/wan_video_dit_moe.py",
             "diffsynth/pipelines/wan_video_recammaster.py")
    return {name:sha256(paths["astra_repo"]/name) for name in names}


def run(config: dict, *, cache_enabled: bool, outdir: Path,
        profiler: bool=False) -> dict:
    validate_settings(config)
    paths = resolve_paths(config)
    import torch
    import numpy as np
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for official Astra")
    # Explicit one-card selection, no automatic GPU selection.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES","")
    if not visible.isdigit() or torch.cuda.device_count() != 1:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to ONE checked idle GPU before running")
    seed = int(config.get("seed",0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()

    run_dir = outdir.resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+
                                  ("_token_" if cache_enabled else "_baseline_")+uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True,exist_ok=False)
    result = {
        "status":"started","mode":"token" if cache_enabled else "baseline",
        "seed":seed,"config":config,"source_sha256":source_provenance(paths),
        "weights":{k:{"file":str(v),"bytes":v.stat().st_size}
                   for k,v in paths.items() if k in ("astra_checkpoint",)},
        "gpu_visible":visible,"profiler":profiler,
        "generated_video":str(run_dir/"generated.mp4"),
    }
    metadata = run_dir/"metadata.json"
    metadata.write_text(json.dumps(result,indent=2,ensure_ascii=False,default=str))
    start = time.perf_counter()
    try:
        with import_upstream(paths["astra_repo"]):
            script = paths["astra_repo"]/"scripts"/"infer_demo.py"
            spec = importlib.util.spec_from_file_location("_astra_token_original_demo",script)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            fn = module.inference_moe_framepack_sliding_window
            kwargs = dict(config["generation"])
            kwargs.update(dit_path=str(paths["astra_checkpoint"]),
                          wan_model_path=str(paths["wan_model_dir"]),
                          condition_image=str(paths["condition_image"]),
                          output_path=str(run_dir/"generated.mp4"))
            unknown=set(kwargs)-set(inspect.signature(fn).parameters)
            if unknown:
                raise ValueError(f"Official Astra signature mismatch: {sorted(unknown)}")
            instrument = TokenInstrumentation(config,cache_enabled)
            factory = module.WanVideoAstraPipeline.from_model_manager
            def attach(*args,**kw):
                pipe=factory(*args,**kw)
                instrument.install(pipe)
                return pipe
            torch.cuda.synchronize()
            with patch.object(module.WanVideoAstraPipeline,
                              "from_model_manager",side_effect=attach):
                if profiler:
                    activities=[torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA]
                    with torch.profiler.profile(activities=activities) as profile:
                        fn(**kwargs)
                    profile.export_chrome_trace(str(run_dir/"profile.json"))
                else:
                    fn(**kwargs)
        torch.cuda.synchronize()
        result["instrumentation"]=instrument.results()
        if instrument.collector is not None:
            instrument.collector.save(run_dir/"teacher_local_proxy.pt")
            result["teacher_proxy_path"]=str(run_dir/"teacher_local_proxy.pt")
        expected_chunks=(int(kwargs["total_frames_to_generate"])+
                         int(kwargs["frames_per_generation"])-1)//int(kwargs["frames_per_generation"])
        if result["instrumentation"]["full_chunks"]!=expected_chunks or not result["instrumentation"]["completed_step_alignment"]:
            raise RuntimeError("DiT invocation count inconsistent with expected AR rollout")
        if not (run_dir/"generated.mp4").is_file():
            raise RuntimeError("Official inference produced no generated.mp4")
        result.update(status="completed",wall_seconds=time.perf_counter()-start,
                      peak_cuda_bytes=int(torch.cuda.max_memory_allocated()))
        return result
    except BaseException as exc:
        result.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        metadata.write_text(json.dumps(result,indent=2,ensure_ascii=False,default=str)+"\n")
        print(f"Metadata: {metadata}")
