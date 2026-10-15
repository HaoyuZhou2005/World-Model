"""Offline config tests, independent of server paths."""
import json
from pathlib import Path
import pytest

from astra_research.config import load_config, fingerprint, resolve_path


def test_rejects_unknown_generation_fields(tmp_path: Path):
    path = tmp_path / "example.json"
    obj = {
        "paths": {},
        "generation": {
            "frames_per_generation": 4,
            "total_frames_to_generate": 8,
            "invented_option": True,
        },
        "cache": {"method": "none"},
        "measurement": {},
    }
    path.write_text(json.dumps(obj), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported generation"):
        load_config(path)


def test_placeholder_fails_closed(tmp_path: Path):
    obj = {"_config_file": str(tmp_path / "config.json")}
    with pytest.raises(ValueError, match="Unresolved"):
        resolve_path("TODO_ABSOLUTE_PATH_TO_ASTRA", obj)


def test_fingerprint_is_stable():
    cfg = {"paths": {}, "generation": {}, "_config_file": "x"}
    assert fingerprint(cfg) == fingerprint({**cfg, "_config_file": "y"})
