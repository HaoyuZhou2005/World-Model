"""In-process invocation of the official Astra inference function.

No fork/patch of upstream model source is committed. A factory patch attaches
hooks after the pipeline is built and BEFORE official inference starts.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import importlib.util
import inspect
import os
from pathlib import Path
import random
import sys
from typing import Any
from unittest.mock import patch

from .instrument import AstraInstrumentation


@contextmanager
def upstream_import_path(root: Path):
    root_s = str(root.resolve())
    inserted = root_s not in sys.path
    if inserted:
        sys.path.insert(0, root_s)
    try:
        yield
    finally:
        if inserted:
            sys.path.remove(root_s)


def load_official_inference(astra_repo: Path) -> Any:
    script = astra_repo / "scripts" / "infer_demo.py"
    spec = importlib.util.spec_from_file_location("_external_astra_infer_demo", script)
    if not spec or not spec.loader:
        raise RuntimeError(f"Unable to load Astra inference module: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = ("inference_moe_framepack_sliding_window", "WanVideoAstraPipeline")
    for name in required:
        if not hasattr(module, name):
            raise RuntimeError(f"Upstream Astra no longer provides {name}; update adapter")
    return module


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_official(
    astra_repo: Path, kwargs: dict[str, Any], config: dict[str, Any], *,
    cache_enabled: bool, trace_path: Path | None = None,
) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Official Astra requires a CUDA-capable GPU; no GPU detected")
    instrumentation = AstraInstrumentation(config, cache_enabled)
    with upstream_import_path(astra_repo):
        module = load_official_inference(astra_repo)
        fn = module.inference_moe_framepack_sliding_window
        signature = inspect.signature(fn)
        unknown = set(kwargs) - set(signature.parameters)
        if unknown:
            raise RuntimeError(f"Upstream Astra inference signature changed: {sorted(unknown)}")

        factory = module.WanVideoAstraPipeline.from_model_manager

        def attach(*args, **keyword_args):
            pipe = factory(*args, **keyword_args)
            instrumentation.install(pipe)
            return pipe

        # This patch only changes the pipeline factory temporarily; it restores
        # the original immediately after inference even if generation fails.
        with patch.object(module.WanVideoAstraPipeline, "from_model_manager", side_effect=attach):
            if trace_path:
                activities = [torch.profiler.ProfilerActivity.CPU,
                              torch.profiler.ProfilerActivity.CUDA]
                profiler = torch.profiler.profile(
                    activities=activities, record_shapes=False,
                    profile_memory=False, with_stack=False)
                with profiler:
                    fn(**kwargs)
                profiler.export_chrome_trace(str(trace_path))
            else:
                fn(**kwargs)

    result = instrumentation.results()
    if result["dit_forward_calls"] == 0:
        raise RuntimeError("No DiT forwards were recorded; integration may be broken")
    if (result["dit_forward_calls"] % (instrumentation.expected_steps * instrumentation.branches)) != 0:
        raise RuntimeError(
            "Observed DiT forward count not divisible by expected steps/CFG branches. "
            "Step mapping invalid; DO NOT report cache results until this is resolved.")
    return result
