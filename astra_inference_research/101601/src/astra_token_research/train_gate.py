"""Offline LOCAL teacher-student gate calibration (not future-safe RL training).

Training the MLP on cached local teacher errors does NOT establish
long-horizon rollout safety. The latter must use real AR student rollouts and
future prediction loss, specified in docs/RESEARCH_PLAN.md.

Dataset .pt expected:
 features [N,F], fresh_residual [N,D], stale_residual [N,D], no model weights.
Examples are generated on the server in a separate, explicitly enabled
teacher collection step. No fabricated dataset or silent random "training".
"""
from __future__ import annotations


def build_mlp(input_dim: int, hidden_dim: int = 64):
    import torch.nn as nn
    if input_dim < 2 or hidden_dim < 2:
        raise ValueError("MLP dimensions too small")
    return nn.Sequential(nn.Linear(input_dim, hidden_dim),
                         nn.SiLU(), nn.Linear(hidden_dim, 1))


def train_local_proxy(data_path, output_path, *, epochs=20, lr=1e-3,
                      budget=0.5, penalty=0.1, seed=0):
    import torch
    torch.manual_seed(seed)
    data = torch.load(data_path, weights_only=True, map_location="cpu")
    features = data["features"].float()
    fresh = data["fresh_residual"].float()
    stale = data["stale_residual"].float()
    if features.ndim != 2 or fresh.shape != stale.shape or fresh.shape[0] != features.shape[0]:
        raise ValueError("Invalid teacher dataset shapes")
    if not 0 < budget < 1:
        raise ValueError("budget must be in (0,1)")
    # Training target measures local residual error only.
    # This is NOT future-rollout loss and not a guarantee of speedup.
    discrepancy = (fresh - stale).square().mean(dim=-1).detach()
    discrepancy = discrepancy / discrepancy.mean().clamp_min(1e-8)
    model = build_mlp(features.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(features).squeeze(-1)
        probs = torch.sigmoid(logits)
        expected_error = ((1-probs)*discrepancy).mean()
        budget_penalty = (probs.mean() - budget).square()
        loss = expected_error + penalty * budget_penalty
        loss.backward()
        optimizer.step()
        history.append({"epoch": epoch, "proxy_loss": float(loss.detach()),
                        "mean_gate": float(probs.detach().mean())})
    torch.save({"state_dict": model.state_dict(), "input_dim": features.shape[1],
                "hidden_dim": 64, "budget": budget, "seed": seed,
                "training_objective": "local_residual_proxy_NOT_future_rollout"}, output_path)
    return history
