"""Train a lightweight token-gate predictor on *measured* oracle costs.

Two explicitly different objectives:
- local: teacher's (full block delta - previous delta)^2; diagnostic proxy
- future: precomputed token counterfactual cost from matched MULTI-CHUNK
  teacher/student rollouts; caller must supply genuine oracle_future_cost.

This code NEVER pretends that local labels are future rollout gradients.
A hard top-k inference selection is not differentiable; supervised future
counterfactual labels are one defensible way to learn a future-risk ranking.
"""
from __future__ import annotations

from pathlib import Path
from .features import FEATURE_NAMES


def build_mlp(input_dim: int, hidden_dim: int = 64):
    import torch.nn as nn
    if input_dim != len(FEATURE_NAMES):
        raise ValueError(f"Expected {len(FEATURE_NAMES)}-dimensional gate features")
    if hidden_dim < 8:
        raise ValueError("hidden_dim>=8 required")
    return nn.Sequential(nn.Linear(input_dim, hidden_dim),
                         nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
                         nn.SiLU(), nn.Linear(hidden_dim, 1))


def train(data_path: str, output_path: str, *, label: str="local",
          budget: float=0.5, epochs: int=40, seed: int=0,
          hidden_dim: int=64, lr:float=1e-3) -> dict:
    import torch
    if label not in ("local","future"):
        raise ValueError("label must be 'local' or 'future'")
    if not 0<budget<1 or epochs<1:
        raise ValueError("0<budget<1 and epochs>=1 required")
    torch.manual_seed(seed)
    data=torch.load(data_path,map_location="cpu",weights_only=True)
    if tuple(data.get("feature_names",()))!=FEATURE_NAMES:
        raise ValueError("Incompatible token gate teacher feature schema")
    features=data["features"].float()
    key="oracle_local_error" if label=="local" else "oracle_future_cost"
    if key not in data:
        raise ValueError(
            f"Teacher dataset has no {key}. Future labels MUST be computed "
            "by actual counterfactual AR rollout, never substituted by local deltas.")
    errors=data[key].float().view(-1)
    if features.ndim!=2 or features.shape[1]!=len(FEATURE_NAMES) or (
        errors.shape[0]!=features.shape[0]):
        raise ValueError("Incorrect oracle dataset size")
    if features.shape[0]<16 or not torch.isfinite(features).all() or (
        not torch.isfinite(errors).all()):
        raise ValueError("Dataset too small or contains nonfinite values")
    if bool((errors<0).any()):
        raise ValueError("Oracle costs must be nonnegative")
    if errors.max()<=errors.min():
        raise ValueError("All oracle scores identical; no discriminative training signal")
    # Train score to predict the highest-risk tokens under a HARD budget.
    # Top-k labels are measured full-reference costs; no fake future loss.
    k=max(1,min(len(errors)-1,round(budget*len(errors))))
    oracle_label=torch.zeros_like(errors)
    oracle_label[errors.topk(k).indices]=1.0
    net=build_mlp(features.shape[1],hidden_dim)
    opt=torch.optim.AdamW(net.parameters(),lr=lr)
    history=[]
    for epoch in range(epochs):
        opt.zero_grad()
        logits=net(features).flatten()
        ce=torch.nn.functional.binary_cross_entropy_with_logits(logits,oracle_label)
        prob=logits.sigmoid()
        budget_cost=(prob.mean()-budget).square()
        loss=ce+0.2*budget_cost
        loss.backward()
        opt.step()
        history.append({
            "epoch":epoch,"loss":float(loss.detach()),
            "predicted_budget":float(prob.detach().mean()),
        })
    objective=("local_residual_proxy_NOT_future_rollout" if label=="local" else
               "future_rollout_counterfactual")
    checkpoint={
        "state_dict":net.state_dict(),
        "input_dim":features.shape[1],
        "hidden_dim":hidden_dim,
        "feature_names":FEATURE_NAMES,
        "training_objective":objective,
        "seed":seed,"target_budget":budget,
        "source_data":str(Path(data_path).resolve()),
    }
    torch.save(checkpoint,output_path)
    return {"trained_examples":len(errors),"label":label,
            "objective":objective,"checkpoint":str(output_path),
            "final_loss":history[-1]["loss"],
            "warning":"Future-safe claims require multi-scene downstream rollout validation."}


def main():
    import argparse,json
    p=argparse.ArgumentParser()
    p.add_argument("--data",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--label",choices=["local","future"],default="local")
    p.add_argument("--budget",type=float,default=0.5)
    p.add_argument("--epochs",type=int,default=40)
    p.add_argument("--seed",type=int,default=0)
    a=p.parse_args()
    print(json.dumps(train(a.data,a.output,label=a.label,budget=a.budget,
                            epochs=a.epochs,seed=a.seed),
                     indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
