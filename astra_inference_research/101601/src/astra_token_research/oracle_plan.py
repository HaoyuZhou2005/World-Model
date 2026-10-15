"""Create explicit one-token AR counterfactual configs from a FULL baseline.

Dry/offline planning only. Each config causes a separate multi-chunk
inference when the researcher later invokes the CLI on an idle GPU.
Sampling uses the ACTUAL FramePack token bounds observed in baseline runs.
"""
from __future__ import annotations

from copy import deepcopy
import argparse
import json
from pathlib import Path


def make_configs(metadata_path, output_directory, *,
                 blocks: list[int], chunks: list[int], steps: list[int],
                 tokens_per_stream: int=2, max_configs: int=64):
    meta=json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if meta.get("status")!="completed" or meta.get("mode")!="baseline":
        raise ValueError("Need a completed no-cache token instrumentation run")
    layouts=meta.get("instrumentation",{}).get("layout_by_chunk",{})
    if not layouts:
        raise ValueError("No FramePack layouts recorded; no safe token indices can be sampled")
    if tokens_per_stream<1 or max_configs<1:
        raise ValueError("Positive token budget and max_configs required")
    out=Path(output_directory).resolve()
    out.mkdir(parents=True,exist_ok=True)
    manifest=[]
    base=meta["config"]
    if not blocks or not chunks or not steps:
        raise ValueError("Choose explicit layer/chunk/denoising-step intervention ranges")
    for chunk in chunks:
        layout=layouts.get(str(chunk))
        if not layout:
            raise ValueError(f"No observed token layout for chunk {chunk}")
        for block in blocks:
            for step in steps:
                if not 1<=step<50:
                    raise ValueError("Intervention step must be within 1..49")
                for name in ("4x","2x","1x","target"):
                    part=layout.get(name)
                    if not part or not part["count"]:
                        continue
                    start,stop=int(part["start"]),int(part["stop"])
                    n=min(tokens_per_stream,stop-start)
                    points=sorted(set(
                        start+(k*(stop-start-1)//max(1,n-1))
                        for k in range(n)))
                    for token in points:
                        if len(manifest)>=max_configs:
                            break
                        conf=deepcopy(base)
                        conf["token_cache"]["policy"]="intervention"
                        conf["token_cache"]["blocks"]=[block]
                        conf["token_cache"]["model_checkpoint"]=None
                        conf["token_cache"]["intervention"]={
                            "chunk":chunk,"step":step,"branch":0,
                            "block":block,"token":token}
                        conf["collection"]={"enabled":False}
                        identifier=(f"c{chunk}_s{step}_b{block}_{name}_tok{token}")
                        path=out/(identifier+".json")
                        path.write_text(json.dumps(conf,indent=2,ensure_ascii=False)+"\n",
                                        encoding="utf-8")
                        manifest.append({"id":identifier,"config":str(path),
                                         "intervention":conf["token_cache"]["intervention"],
                                         "expected_mode":"token"})
                    if len(manifest)>=max_configs:
                        break
                if len(manifest)>=max_configs:
                    break
            if len(manifest)>=max_configs:
                break
        if len(manifest)>=max_configs:
            break
    file=out/"oracle_manifest.json"
    file.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",
                    encoding="utf-8")
    return manifest


def main():
    p=argparse.ArgumentParser(description="Design one-token stale ablation configs")
    p.add_argument("--baseline-metadata",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--blocks",nargs="+",type=int,required=True)
    p.add_argument("--chunks",nargs="+",type=int,required=True)
    p.add_argument("--steps",nargs="+",type=int,required=True)
    p.add_argument("--tokens-per-stream",type=int,default=2)
    p.add_argument("--max-configs",type=int,default=32)
    a=p.parse_args()
    results=make_configs(a.baseline_metadata,a.out,blocks=a.blocks,
                         chunks=a.chunks,steps=a.steps,
                         tokens_per_stream=a.tokens_per_stream,
                         max_configs=a.max_configs)
    print(f"{len(results)} separate FULL rollouts planned. Nothing was executed.")


if __name__=="__main__":
    main()
