"""Offline research-infrastructure tests, no GPU needed."""
from copy import deepcopy
from pathlib import Path

import pytest

from astra_research.benchmark import compare_pair, paired_repeats
from astra_research.experiment_plan import generate


def record(mode, time, gpu_ms, cache_cfg=None):
    return {
        "_source": "synthetic",
        "status": "completed", "mode": mode, "seed": 0,
        "astra_revision": "UNKNOWN_GIT_REVISION",
        "upstream_files_sha256": {"scripts/infer_demo.py": "deadbeef"},
        "weights_sha256": {"astra_ckpt": "sha-verified-in-synthetic-fixture"},
        "elapsed_seconds": time, "torch_profiler": False,
        "effective_config": {
            "paths": {"astra_repo": "/stub", "astra_checkpoint": "/stub/weights",
                      "wan_model_dir": "/wan", "condition_image": "/image"},
            "generation": {"prompt": "synthetic"}, "cache": cache_cfg or {},
            "measurement": {"expected_denoising_steps": 50},
        },
        "instrumentation": {
            "dit_forward_calls": 150,
            "dit_cuda_time_ms_sum": gpu_ms,
            "cache": {},
        },
    }


def test_config_differences_only_cache_are_comparable():
    a = record("baseline", 200, 130000)
    b = record("cache", 150, 65000, {"method": "scheduled"})
    result = compare_pair(a, b)
    assert result["verified_comparable"]
    assert result["wall_speedup_x"] == pytest.approx(4 / 3)
    assert result["dit_speedup_x"] == pytest.approx(2.0)


def test_mismatch_seed_rejected():
    a = record("baseline", 200, 130000)
    b = record("cache", 150, 65000)
    b["seed"] = 999
    assert "seed" in compare_pair(a, b)["mismatches"]
    with pytest.raises(ValueError):
        paired_repeats([a, a], [b, b])


def test_unknown_upstream_needs_hash():
    a = record("baseline", 200, 130000)
    b = record("cache", 150, 65000)
    del b["upstream_files_sha256"]
    assert not compare_pair(a, b)["verified_comparable"]


def test_paired_summary_uses_median_of_pairs():
    a = [record("baseline", 200, 130000), record("baseline", 240, 140000)]
    b = [record("cache", 160, 65000), record("cache", 200, 70000)]
    result = paired_repeats(a, b)
    assert result["n_pairs"] == 2
    assert result["wall_speedup_median_x"] == pytest.approx(1.225)


def test_generated_configs_do_not_touch_gpu(tmp_path: Path):
    template = {
        "seed": 0, "paths": {"astra_repo": "TODO_PATH"},
        "generation": {"frames_per_generation": 8, "total_frames_to_generate": 24},
        "cache": {}, "measurement": {},
    }
    manifest = generate(template, tmp_path / "configs", num_blocks=30)
    assert "progressive_k" in manifest
    import json
    progressive = json.loads(Path(manifest["progressive_k"]["config"]).read_text())
    assert progressive["cache"]["warmup_by_chunk"] == {"0": 1, "1": 2, "2": 4}
    baseline = json.loads(Path(manifest["baseline"]["config"]).read_text())
    assert len(baseline["cache"]["block_indices"]) == 30



def test_sweep_keeps_original_relative_path_semantics(tmp_path: Path):
    original_dir = tmp_path / "configs"
    original_dir.mkdir()
    template = {
        "seed": 0,
        "paths": {
            "astra_repo": "../../external/Astra",
            "identity_manifest": "identity_local.json",
            "condition_image": "TODO_CONDITION_IMAGE",
            "output_root": "../runs",
        },
        "generation": {"frames_per_generation": 8, "total_frames_to_generate": 24},
        "cache": {}, "measurement": {},
    }
    import json
    manifest = generate(template, original_dir / "sweep_local",
                        num_blocks=30, names=["baseline"],
                        template_base=original_dir)
    data = json.loads(Path(manifest["baseline"]["config"]).read_text())
    assert data["paths"]["astra_repo"] == str((original_dir / "../../external/Astra").resolve())
    assert data["paths"]["identity_manifest"] == str((original_dir / "identity_local.json").resolve())
    assert data["paths"]["output_root"] == str((original_dir / "../runs").resolve())
    assert data["paths"]["condition_image"] == "TODO_CONDITION_IMAGE"
