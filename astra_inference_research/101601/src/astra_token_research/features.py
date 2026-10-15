"""Fixed 9-feature schema shared by online gate and offline teacher collector.

Short horizon: denoising step and token activation change since prior step.
Long horizon: chunk index plus FramePack stream identity. These are conditioning
signals, NOT a calibrated forecast of downstream rollout error.
"""
from __future__ import annotations

FEATURE_NAMES = ("relative_input_change","previous_delta_abs",
                 "age_normalized","step_normalized","chunk_normalized",
                 "is_4x","is_2x","is_1x","is_target")


def token_features(x, previous_input, previous_delta, last_refresh,
                   layout, *, chunk: int, step: int, steps: int):
    import torch
    if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] != layout.total:
        raise ValueError("B=1 [1,S,D] and exact packed layout required")
    current = x.detach().float()[0]
    before = previous_input.detach().float()[0]
    delta = previous_delta.detach().float()[0]
    change = (current - before).abs().mean(-1) / before.abs().mean(-1).clamp_min(1e-5)
    dnorm = delta.abs().mean(-1)
    age = (step-last_refresh).clamp_min(0).float()/max(1,steps)
    idx = torch.arange(layout.total, device=x.device)
    base = [change, dnorm, age,
            torch.full_like(change,step/max(1,steps-1)),
            torch.full_like(change,chunk/(1+chunk))]
    for name in ("4x","2x","1x","target"):
        stream = layout.get(name)
        base.append(((idx>=stream.start)&(idx<stream.stop)).float())
    return torch.stack(base,dim=-1)  # [S,9], float32
