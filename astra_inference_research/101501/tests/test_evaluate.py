"""Video metrics use synthetic numpy frame sequences, no GPU or video codec needed."""
import pytest

np = pytest.importorskip("numpy")

import astra_research.evaluate as evaluate


def test_chunk_bounds_and_mae(monkeypatch):
    rgb = lambda value: np.full((2, 2, 3), value, dtype=np.float32)
    reference = [rgb(0), rgb(10), rgb(10), rgb(20), rgb(20)]
    candidate = [rgb(0), rgb(10), rgb(12), rgb(23), rgb(23)]
    def fake_frames(path):
        yield from (reference if str(path) == "baseline" else candidate)
    monkeypatch.setattr(evaluate, "_frames", fake_frames)
    result = evaluate.evaluate_pair("baseline", "cache",
                                    rgb_frames_per_chunk=2,
                                    n_chunks=2, condition_frames=1)
    assert result["per_chunk"][0]["mae_to_no_cache"] == 1.0
    assert result["per_chunk"][1]["mae_to_no_cache"] == 3.0


def test_mismatched_frame_count_is_not_silently_truncated(monkeypatch):
    monkeypatch.setattr(evaluate, "_frames",
                        lambda path: iter([np.zeros((2, 2, 3))] * (4 if str(path)=="a" else 5)))
    with pytest.raises(ValueError, match="Expected exactly"):
        evaluate.evaluate_pair("a", "b", rgb_frames_per_chunk=2,
                               n_chunks=2, condition_frames=1)


def test_identical_video_json_is_valid(monkeypatch):
    monkeypatch.setattr(evaluate, "_frames",
                        lambda path: iter([np.zeros((2, 2, 3))] * 3))
    result = evaluate.evaluate_pair("a", "b", rgb_frames_per_chunk=2, n_chunks=1)
    assert result["per_chunk"][0]["perfect_match"]
    import json
    json.dumps(result, allow_nan=False)
