"""Synthetic tests for future counterfactual dataset I/O; no GPU/MP4 needed."""
import json
from pathlib import Path
import pytest
torch=pytest.importorskip("torch")
np=pytest.importorskip("numpy")

from astra_token_research import counterfactual


def test_one_token_future_risk_label_from_real_comparison_structure(tmp_path, monkeypatch):
    # MP4 decoder mocked; REAL experiments must use existing generated videos.
    def frames(path):
        is_baseline="baseline.mp4" in str(path)
        for index in range(5):
            value=0 if is_baseline or index<3 else 3
            yield np.full((2,2,3),value,dtype=np.float32)
    monkeypatch.setattr(counterfactual,"_read_frames",frames)
    gen={"frames_per_generation":2,"total_frames_to_generate":4,
         "prompt":"same"}
    common={"status":"completed","seed":1,"source_sha256":{"a":"b"},
            "weights":{"ckpt":{"file":"same","bytes":100}},
            "config":{"generation":gen,"token_cache":{}}}
    base={**common,"mode":"baseline","generated_video":"baseline.mp4"}
    intervention={
        **common,"mode":"token","generated_video":"modified.mp4",
        "config":{"generation":gen,"token_cache":{"policy":"intervention"}},
        "instrumentation":{"cache":{"intervention_observations":[{
            "chunk":0,"step":20,"branch":0,"block":4,
            "token":7,"stream":"target","feature":[0.1]*9
        }]}}
    }
    baseline_path=tmp_path/"base.json"
    modified_path=tmp_path/"modified.json"
    baseline_path.write_text(json.dumps(base))
    modified_path.write_text(json.dumps(intervention))
    result_path=tmp_path/"future.pt"
    n=counterfactual.construct_counterfactual_dataset(
        baseline_path,[modified_path],result_path,
        rgb_per_chunk=2,condition_frames=1)
    assert n==1
    out=torch.load(result_path,weights_only=True)
    assert out["features"].shape==(1,9)
    assert out["oracle_future_cost"][0].item()>0
    assert out["label_type"]=="matched_one_token_stale_AR_future_RGB_MSE_proxy"


def test_counterfactual_rejects_different_source_weights(tmp_path):
    reference={"status":"completed","mode":"baseline","seed":0,
               "source_sha256":{},"weights":{},
               "config":{"generation":{"frames_per_generation":2,"total_frames_to_generate":4}},
               "generated_video":"none.mp4"}
    intervention={**reference,"mode":"token","seed":1,"instrumentation":{"cache":{}}}
    a=tmp_path/"a.json"
    b=tmp_path/"b.json"
    a.write_text(json.dumps(reference))
    b.write_text(json.dumps(intervention))
    with pytest.raises(ValueError,match="seed"):
        counterfactual.construct_counterfactual_dataset(
            a,[b],tmp_path/"out.pt",rgb_per_chunk=2,condition_frames=1)
