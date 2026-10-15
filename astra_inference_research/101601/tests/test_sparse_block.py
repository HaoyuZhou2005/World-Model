"""CPU contracts: selected-query mathematics, per-token splice, chunk invalidation.

These tests use a small Astra-shaped fake, NOT real pretrained model weights.
"""
import pytest
torch=pytest.importorskip("torch")
import torch.nn as nn
import torch.nn.functional as F

from astra_token_research.sparse_block import selected_block_rows
from astra_token_research.layout import PackedLayout
from astra_token_research.gate import GateConfig, TokenGate
from astra_token_research.token_cache import TokenResidualCache


def modulate(x, shift, scale):
    return x*(1+scale)+shift


def rope_apply(x, freqs, num_heads):
    return x  # Identity ONLY for this algebraic CPU fake.


class FakeAttention(nn.Module):
    def forward(self, q, k, v):
        # 8-dim model / 2 heads
        def split(a):
            return a.view(a.shape[0],a.shape[1],2,4).transpose(1,2)
        r=F.scaled_dot_product_attention(split(q),split(k),split(v))
        return r.transpose(1,2).flatten(2)


class FakeSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads=2
        self.q=nn.Linear(8,8)
        self.k=nn.Linear(8,8)
        self.v=nn.Linear(8,8)
        self.o=nn.Linear(8,8)
        self.norm_q=nn.Identity()
        self.norm_k=nn.Identity()
        self.attn=FakeAttention()
    def forward(self,x,freqs):
        q=rope_apply(self.norm_q(self.q(x)),freqs,2)
        k=rope_apply(self.norm_k(self.k(x)),freqs,2)
        return self.o(self.attn(q,k,self.v(x)))


class FakeCrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q=nn.Linear(8,8)
        self.k=nn.Linear(8,8)
        self.v=nn.Linear(8,8)
        self.o=nn.Linear(8,8)
        self.attn=FakeAttention()
    def forward(self,x,context):
        return self.o(self.attn(self.q(x),self.k(context),self.v(context)))


class FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn=FakeSelfAttention()
        self.cross_attn=FakeCrossAttention()
        self.norm1=nn.LayerNorm(8,elementwise_affine=False)
        self.norm2=nn.LayerNorm(8,elementwise_affine=False)
        self.norm3=nn.LayerNorm(8)
        self.projector=nn.Linear(8,8)
        self.ffn=nn.Sequential(nn.Linear(8,16),nn.GELU(),nn.Linear(16,8))
        self.modulation=nn.Parameter(torch.zeros(1,6,8))
        self.use_moe=False

    def forward(self,x,context,cam_emb,t_mod,freqs,modality_inputs=None,
                router_weights=None,router_indices=None):
        shift,scale,gate,mlps,mlpc,mlpg=(self.modulation+t_mod).chunk(6,dim=1)
        input_x=modulate(self.norm1(x),shift,scale)
        attention=self.self_attn(input_x,freqs)
        x=x+gate*self.projector(attention)
        x=x.to(self.norm3.weight.dtype)
        x=x+self.cross_attn(self.norm3(x),context)
        return x+mlpg*self.ffn(modulate(self.norm2(x),mlps,mlpc))


def setup():
    torch.manual_seed(13)
    block=FakeBlock()
    with torch.no_grad():
        block.modulation[:,2].fill_(0.5)
        block.modulation[:,5].fill_(0.5)
    x=torch.randn(1,9,8)
    context=torch.randn(1,4,8)
    modulation=torch.randn(1,6,8)*0.1
    freqs=torch.zeros(1,9,1,4)
    return block,x,context,modulation,freqs


def test_query_sparse_matches_selected_dense_rows():
    b,x,ctx,t,f=setup()
    indices=torch.tensor([0,3,6,8],dtype=torch.long)
    with torch.no_grad():
        dense=b(x,ctx,None,t,f)
        sparse=selected_block_rows(b,x,indices,ctx,None,t,f)
    torch.testing.assert_close(sparse,dense.index_select(1,indices),
                               rtol=1e-5,atol=1e-5)


def test_cache_keeps_selected_rows_fresh_and_stale_others():
    b,x,ctx,t,f=setup()
    layout=PackedLayout.from_shapes({"target":(1,3,3)},9)
    cfg=GateConfig(ratios={"target":1/3},min_per_stream=0,
                   max_age=2,first_full_steps=1,force_last_step_full=False)
    cache=TokenResidualCache(TokenGate(cfg,10),selected_blocks=[0],enabled=True)
    args=(ctx,None,t,f)
    with torch.no_grad():
        y0=cache.run(b,0,x,b.forward,args,{},layout=layout,chunk=0,step=0,branch=0)
        x1=x.clone()
        x1[0,4] += 2.0
        expected=b(x1,*args)
        y1=cache.run(b,0,x1,b.forward,args,{},layout=layout,chunk=0,step=1,branch=0)
    torch.testing.assert_close(y1[:,4],expected[:,4],rtol=1e-5,atol=1e-5)
    assert 0 < cache.stale_rows < cache.total_query_rows
    assert cache.partial_calls==1
    # Cache entries MUST clear between different AR chunks.
    with torch.no_grad():
        cache.run(b,0,x1,b.forward,args,{},layout=layout,chunk=1,step=0,branch=0)
    assert cache.current_chunk==1
    assert cache.full_calls==2


def test_unsupported_batch_rejected():
    b,x,ctx,t,f=setup()
    with pytest.raises(ValueError,match="B=1"):
        selected_block_rows(b,x.expand(2,-1,-1),torch.tensor([1]),ctx,None,t,f)
