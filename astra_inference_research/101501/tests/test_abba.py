"""ABBA planning does not need GPU, model weights or subprocess invocation."""
import pytest

from astra_research.abba import abba_sequence, run


def test_abba_balanced_order(tmp_path):
    sequence = abba_sequence("baseline.json", "cache.json", 2)
    assert len(sequence) == 8
    assert [x["mode"] for x in sequence[:4]] == [
        "baseline", "cache", "cache", "baseline"]
    assert [x["mode"] for x in sequence[4:]] == [
        "baseline", "cache", "cache", "baseline"]


def test_no_explicit_execute_means_no_gpu_job(tmp_path):
    result = run(abba_sequence("baseline.json", "cache.json", 1),
                 log_dir=tmp_path / "logs", execute=False)
    assert result["dry_run"]
    assert not (tmp_path / "logs").exists()


def test_zero_cycles_rejected():
    with pytest.raises(ValueError):
        abba_sequence("baseline", "cache", 0)
