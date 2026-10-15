"""Build *measured* future-risk labels from FULL baseline and interventions.

Input interventions must have been run with one-token stale mode in
token_cache.py: all OTHER token rows and blocks were full-computed.
We compare generated RGB frames by requested chunk boundaries; this gives a
future *pixel-deviation* proxy, not ground-truth realism or camera fidelity.

One record = one intervention's measured downstream rollout, NOT fabricated
per-token labels from local L1 error. Interventions are costly: each
requires a separate full-length autoregressive inference.
"""
from __future__ import annotations

from itertools import zip_longest
import json
from pathlib import Path


def _read_frames(path):
    import imageio.v2 as imageio
    import numpy as np
    reader=imageio.get_reader(str(path))
    try:
        while True:
            try:
                yield np.asarray(reader.get_next_data(),dtype=np.float32)
            except (IndexError,StopIteration):
                break
    finally:
        reader.close()


def future_chunk_errors(baseline_mp4, modified_mp4, *,
                        chunks, rgb_per_chunk, condition_frames=1):
    import numpy as np
    expected=condition_frames + chunks*rgb_per_chunk
    totals=[0.0]*chunks
    counts=[0]*chunks
    seen=0
    sentinel=object()
    for a,b in zip_longest(_read_frames(baseline_mp4),_read_frames(modified_mp4),
                           fillvalue=sentinel):
        if a is sentinel or b is sentinel:
            raise ValueError("Videos must have exactly matching numbers of frames")
        if a.shape!=b.shape:
            raise ValueError("Different frame shapes are not comparable")
        if seen>=expected:
            raise ValueError("Video has more frames than explicitly expected")
        if seen>=condition_frames:
            j=(seen-condition_frames)//rgb_per_chunk
            difference=(a.astype(np.float64)-b.astype(np.float64))
            totals[j]+=float(np.square(difference).sum())
            counts[j]+=int(difference.size)
        seen+=1
    if seen!=expected:
        raise ValueError(f"Expected {expected} frames, decoded {seen}")
    return [totals[i]/counts[i] for i in range(chunks)]


def _read_metadata(path):
    f=Path(path)
    if f.is_dir():
        f=f/"metadata.json"
    data=json.loads(f.read_text(encoding="utf-8"))
    if data.get("status")!="completed":
        raise ValueError(f"Run is incomplete: {f}")
    return data


def construct_counterfactual_dataset(
    baseline_metadata, intervention_metadata, out_path, *,
    rgb_per_chunk: int, condition_frames: int,
    horizon_weight: float=1.0,
):
    import torch
    if horizon_weight<0 or rgb_per_chunk<1 or condition_frames<0:
        raise ValueError("Invalid frame or horizon weighting parameters")
    reference=_read_metadata(baseline_metadata)
    if reference["mode"]!="baseline":
        raise ValueError("Expected a baseline full-compute run")
    generation=reference["config"]["generation"]
    target=int(generation["total_frames_to_generate"])
    per=int(generation["frames_per_generation"])
    if target%per:
        raise ValueError("Counterfactual oracle requires equal chunk lengths")
    chunks=target//per
    examples=[]
    for path in intervention_metadata:
        trial=_read_metadata(path)
        if trial["mode"]!="token":
            raise ValueError("Counterfactual inference must be in token mode")
        for name in ("seed","source_sha256","weights"):
            if trial.get(name)!=reference.get(name):
                raise ValueError(f"Baseline and intervention mismatch: {name}")
        if trial["config"]["generation"]!=generation:
            raise ValueError("Different generation arguments/actions invalidate comparison")
        obs=trial["instrumentation"]["cache"].get("intervention_observations",[])
        if len(obs)!=1:
            raise ValueError("Exactly one stale token per counterfactual trial required")
        observation=obs[0]
        if trial["config"]["token_cache"].get("policy")!="intervention":
            raise ValueError("Intervention config must explicitly declare its causal policy")
        chunk=int(observation["chunk"])
        if not 0<=chunk<chunks:
            raise ValueError("Intervention chunk outside generation horizon")
        errors=future_chunk_errors(
            reference["generated_video"],trial["generated_video"],
            chunks=chunks,rgb_per_chunk=rgb_per_chunk,
            condition_frames=condition_frames)
        # Long-horizon score includes the intervention chunk and all LATER
        # chunks, downweighting no future effects by simply not including
        # earlier chunks which the intervention could not have caused.
        weights=[1.0+horizon_weight*(i-chunk)/max(1,chunks-1-chunk)
                 for i in range(chunk,chunks)]
        risk=sum(weights[j]*errors[chunk+j] for j in range(len(weights)))/sum(weights)
        examples.append({
            "feature":observation["feature"],
            "oracle_future_cost":max(0.0,float(risk)),
            "per_chunk_mse":errors,
            "intervention":{k:observation[k] for k in
                            ("chunk","step","branch","block","token","stream")},
            "source_metadata":str(path),
        })
    if not examples:
        raise ValueError("No counterfactual trials")
    features=torch.tensor([e["feature"] for e in examples],dtype=torch.float32)
    if features.ndim!=2 or features.shape[1]!=9:
        raise ValueError("Incompatible gate feature schema")
    from .features import FEATURE_NAMES
    torch.save({
        "features":features,
        "oracle_future_cost":torch.tensor(
            [e["oracle_future_cost"] for e in examples],dtype=torch.float32),
        "feature_names":FEATURE_NAMES,
        "label_type":"matched_one_token_stale_AR_future_RGB_MSE_proxy",
        "source_baseline":str(baseline_metadata),
        "horizon_weight":horizon_weight,
        "examples":examples,
        "warning":(
            "Pixel deviation to baseline is a counterfactual FUTURE difference, "
            "not ground-truth predictive accuracy or proof of future safety."
        ),
    },str(out_path))
    return len(examples)


def main():
    import argparse
    parser=argparse.ArgumentParser(description="Generate measured future-risk teacher labels")
    parser.add_argument("--baseline",required=True)
    parser.add_argument("--interventions",required=True,nargs="+")
    parser.add_argument("--rgb-per-chunk",type=int,required=True)
    parser.add_argument("--condition-frames",type=int,default=1)
    parser.add_argument("--out",required=True)
    a=parser.parse_args()
    n=construct_counterfactual_dataset(
        a.baseline,a.interventions,a.out,
        rgb_per_chunk=a.rgb_per_chunk,condition_frames=a.condition_frames)
    print(json.dumps({"examples":n,"dataset":a.out},indent=2))


if __name__=="__main__":
    main()
