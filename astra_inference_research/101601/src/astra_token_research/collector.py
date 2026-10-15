"""Observe FULL-compute teacher residuals; create local gate proxy dataset.

No labels are described as 'future-safe'. The target is the change in output
from reusing previous-step residual instead of freshly computing this block.
Downstream error must be evaluated through separate future autoregressive
rollouts and/or counterfactual labels before publication.
"""
from __future__ import annotations
import torch

from .features import FEATURE_NAMES,token_features


class TeacherCollector:
    def __init__(self, samples_per_step: int = 128):
        if samples_per_step < 1:
            raise ValueError("samples_per_step >=1 required")
        self.samples_per_step=samples_per_step
        self.last:dict[tuple[int,int],tuple] = {}
        self.chunk=-1
        self.features=[]
        self.errors=[]
        self.rows=0

    def observe(self,x,full_output,*, layout,chunk,step,branch,block_id,steps):
        if self.chunk!=chunk:
            self.last.clear()
            self.chunk=chunk
        key=(branch,block_id)
        delta=(full_output-x).detach()
        prior=self.last.get(key)
        if prior is not None:
            old_x, old_delta, old_step, old_signature = prior
            if old_x.shape==x.shape and old_signature==layout.signature():
                n=x.shape[1]
                features=token_features(x,old_x,old_delta,
                    torch.full((n,),old_step,device=x.device,dtype=torch.long),
                    layout,chunk=chunk,step=step,steps=steps)
                # Deterministic even sampling, no full-D weight data stored.
                count=min(n,self.samples_per_step)
                choice=torch.linspace(0,n-1,count,device=x.device).long()
                target=((delta-old_delta).float().square().mean(-1)[0])
                self.features.append(features.index_select(0,choice).cpu())
                self.errors.append(target.index_select(0,choice).cpu())
                self.rows+=count
        self.last[key]=(x.detach().clone(),delta.clone(),step,layout.signature())

    def save(self,path):
        if not self.features:
            raise RuntimeError("No teacher examples collected; check number of steps")
        torch.save({
            "features":torch.cat(self.features),
            "oracle_local_error":torch.cat(self.errors),
            "feature_names":FEATURE_NAMES,
            "label_type":"local_block_residual_MSE_NOT_future_rollout",
            "num_records":self.rows,
        },path)
