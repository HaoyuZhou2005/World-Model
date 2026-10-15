"""Token selection with separate SHORT denoising and LONG AR-horizon signals.

This is a heuristic/oracle research controller. Random MLP weights are NEVER
called a learned gate. A trained gate can be loaded explicitly after offline
teacher collection; until then use change-based scores as a diagnostic.

A frame-group coarse stage may force all queries within a group to refresh;
fine stage selects per-token Q rows based on budget and score.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from .layout import PackedLayout


@dataclass
class GateConfig:
    ratios: dict[str, float]
    coarse_threshold: float | None = None
    min_per_stream: int = 1
    max_age: int = 1
    first_full_steps: int = 1
    force_last_step_full: bool = True
    horizon_boost: float = 0.0

    def validate(self):
        if not 0 <= self.horizon_boost <= 10:
            raise ValueError("horizon_boost must be in [0,10]")
        if self.max_age < 1 or self.first_full_steps < 1:
            raise ValueError("max_age>=1 and first_full_steps>=1 required")
        if self.min_per_stream < 0:
            raise ValueError("min_per_stream cannot be negative")
        if self.coarse_threshold is not None and self.coarse_threshold < 0:
            raise ValueError("coarse_threshold must be >= 0")
        for name in ("4x", "2x", "1x", "target"):
            value = self.ratios.get(name, 1.0)
            if not 0 <= value <= 1:
                raise ValueError(f"Invalid refresh ratio {name}={value}")


class TokenGate:
    def __init__(self, config: GateConfig, num_steps: int, *, policy: str="change_topk",
                 model_checkpoint: str | None = None):
        config.validate()
        if num_steps < 2:
            raise ValueError("num_steps must be >=2")
        self.config = config
        self.num_steps = num_steps
        if policy not in ("change_topk", "learned"):
            raise ValueError(f"Unknown token gate policy: {policy}")
        self.policy=policy
        self.net=None
        if policy=="learned":
            if not model_checkpoint:
                raise ValueError("Learned token gate requires a trained checkpoint")
            import torch
            from .train_gate import build_mlp
            from .features import FEATURE_NAMES
            state=torch.load(model_checkpoint,map_location="cpu",weights_only=True)
            if state.get("feature_names")!=FEATURE_NAMES:
                raise ValueError("Learned gate feature schema mismatch")
            if state.get("training_objective") not in (
                "local_residual_proxy_NOT_future_rollout",
                "future_rollout_counterfactual"):
                raise ValueError("Unrecognized gate training objective")
            self.net=build_mlp(int(state["input_dim"]),int(state["hidden_dim"]))
            self.net.load_state_dict(state["state_dict"], strict=True)
            self.net.eval()

    def select(self, x, prev_x, last_refresh, layout: PackedLayout,
               *, chunk: int, step: int, scores=None, previous_delta=None):
        import torch
        if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] != layout.total:
            raise ValueError("Only B=1 packed tokens [1,S,D] supported")
        if last_refresh.shape != (layout.total,):
            raise ValueError("Per-token age vector must have shape [S]")
        if (step < self.config.first_full_steps or
                (self.config.force_last_step_full and step == self.num_steps-1)):
            return torch.ones(layout.total, dtype=torch.bool, device=x.device)
        if prev_x is None or prev_x.shape != x.shape:
            return torch.ones(layout.total, dtype=torch.bool, device=x.device)
        # "change" is a latent activation score, NOT RGB frame difference.
        if scores is None:
            if self.policy=="learned":
                if previous_delta is None:
                    raise ValueError("Learned gate needs prior cached residuals")
                from .features import token_features
                feats=token_features(x,prev_x,previous_delta,last_refresh,
                    layout,chunk=chunk,step=step,steps=self.num_steps)
                with torch.no_grad():
                    self.net=self.net.to(x.device)
                    scores=self.net(feats).flatten().sigmoid()
            else:
                difference = (x.detach().float() - prev_x.float()).abs().mean(-1)[0]
                scale = prev_x.float().abs().mean(-1)[0].clamp_min(1e-5)
                scores = difference / scale
        if scores.shape != (layout.total,):
            raise ValueError("Gate scores must be one scalar per token")
        # Short horizon: feature change/age. Long horizon: explicit
        # chunk-conditioned refresh BUDGET, not a useless global score scale.
        # This is a heuristic horizon proxy, NOT a learned future-risk signal.
        scores = scores.float()
        horizon_factor = 1 + self.config.horizon_boost * chunk / max(1, chunk+1)
        result = last_refresh < 0
        result = result | ((step - last_refresh) > self.config.max_age)
        for stream in layout.streams:
            if stream.count == 0:
                continue
            local = scores[stream.start:stream.stop]
            effective_ratio = min(1.0, self.config.ratios.get(stream.name, 1.0) * horizon_factor)
            keep = max(self.config.min_per_stream,
                       int(round(stream.count * effective_ratio)))
            keep = min(stream.count, keep)
            if keep:
                winners = local.topk(keep, sorted=False).indices + stream.start
                result[winners] = True
            # Coarse frame-group safeguard: high-change groups force recompute.
            if self.config.coarse_threshold is not None:
                for a, b in stream.frame_slices():
                    group_score = scores[a:b].mean()
                    if bool((group_score > self.config.coarse_threshold).item()):
                        result[a:b] = True
        return result
