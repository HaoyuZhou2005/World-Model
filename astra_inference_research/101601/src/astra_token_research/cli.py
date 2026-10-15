"""Simple, fail-closed CLI for proposal-scale token reuse research.

Run on the already configured Astra CUDA research server. Unknown model file
locations remain TODO; no online experiment is triggered implicitly.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path

from .runner import resolve_paths, run, validate_settings, source_provenance


def load(path):
    file=Path(path).expanduser().resolve()
    content=json.loads(file.read_text(encoding="utf-8"))
    validate_settings(content)
    paths=content["paths"]
    for key,value in list(paths.items()):
        if not isinstance(value,str) or "TODO_" in value:
            continue
        p=Path(value).expanduser()
        if not p.is_absolute():
            paths[key]=str((file.parent/p).resolve())
    # Explicit learned MLP checkpoint and optional local data paths.
    value=content["token_cache"].get("model_checkpoint")
    if value and "TODO_" not in value:
        p=Path(value).expanduser()
        if not p.is_absolute():
            content["token_cache"]["model_checkpoint"]=str((file.parent/p).resolve())
    return content


def main():
    p=argparse.ArgumentParser(description="101601 token-level Astra experiments")
    sub=p.add_subparsers(dest="cmd",required=True)
    doctor=sub.add_parser("doctor",help="Preflight source/model paths without running GPU")
    doctor.add_argument("--config",required=True)
    experiment=sub.add_parser("run")
    experiment.add_argument("--config",required=True)
    experiment.add_argument("--mode",choices=("baseline","token","teacher"),required=True)
    experiment.add_argument("--out",default="runs")
    experiment.add_argument("--dry-run",action="store_true")
    experiment.add_argument("--profiler",action="store_true",
                            help="Do NOT use profiler runtimes for speedup tables")
    tr=sub.add_parser("train-gate",help="Train a local/future-labeled MLP, offline")
    tr.add_argument("--data",required=True)
    tr.add_argument("--output",required=True)
    tr.add_argument("--label",choices=("local","future"),default="local")
    tr.add_argument("--budget",type=float,default=0.5)
    tr.add_argument("--epochs",type=int,default=40)
    args=p.parse_args()
    if args.cmd=="train-gate":
        from .train_gate import train
        data=train(args.data,args.output,label=args.label,
                   budget=args.budget,epochs=args.epochs)
    else:
        conf=load(args.config)
        if args.cmd=="doctor":
            paths=resolve_paths(conf)
            data={"status":"paths_ok","paths":{k:str(v) for k,v in paths.items()},
                  "source_sha256":source_provenance(paths),
                  "note":"No checkpoint loaded; no GPU test executed."}
        else:
            conf=deepcopy(conf)
            if args.mode=="teacher":
                conf["collection"]={"enabled":True,
                    "samples_per_step":int(conf.get("collection",{}).get("samples_per_step",128))}
            else:
                conf["collection"]={"enabled":False}
            if args.dry_run:
                data={"dry_run":True,"mode":args.mode,"config":conf,
                      "note":"No files/checkpoint/GPU inspected."}
            else:
                data=run(conf,cache_enabled=args.mode=="token",
                         outdir=Path(args.out),profiler=args.profiler)
    print(json.dumps(data,ensure_ascii=False,indent=2,default=str))


if __name__=="__main__":
    main()
