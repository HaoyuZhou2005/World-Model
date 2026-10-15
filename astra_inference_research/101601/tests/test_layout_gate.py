import pytest
torch=pytest.importorskip("torch")

from astra_token_research.layout import PackedLayout,LayoutRecorder
from astra_token_research.gate import GateConfig, TokenGate


def layout():
    return PackedLayout.from_shapes(
        {"4x":(1,1,2),"2x":(1,1,3),"1x":(1,2,2),"target":(2,1,2)}, 13)


def test_actual_framepack_order_and_frame_slices():
    l=layout()
    assert [s.name for s in l.streams]==["4x","2x","1x","target"]
    assert [(s.start,s.stop) for s in l.streams]==[(0,2),(2,5),(5,9),(9,13)]
    assert l.get("target").frame_slices()==[(9,11),(11,13)]


def test_packing_mismatch_is_fatal():
    with pytest.raises(ValueError):
        PackedLayout.from_shapes({"target":(2,2,2)},7)


def test_high_change_scores_refresh_token_rows_not_whole_layer():
    l=layout()
    x=torch.ones(1,13,4)
    prev=torch.ones_like(x)
    x[0,10]=10.
    c=GateConfig(ratios={"4x":0.0,"2x":0.0,"1x":0.0,"target":0.25},
                 min_per_stream=0,max_age=2,force_last_step_full=False)
    mask=TokenGate(c,num_steps=10).select(
        x,prev,torch.zeros(13,dtype=torch.long),l,chunk=1,step=1)
    assert bool(mask[10])
    assert int(mask.sum())==1


def test_per_stream_budget_is_never_reinterpreted_as_block_schedule():
    l=layout()
    x=torch.ones(1,13,4)
    cfg=GateConfig(ratios={"4x":0.5,"2x":0.33,"1x":0.25,"target":0.5},
                   min_per_stream=1,max_age=3,force_last_step_full=False)
    mask=TokenGate(cfg,10).select(x,x,torch.zeros(13,dtype=torch.long),l,
                                  chunk=0,step=1)
    assert int(mask.sum())==5
    assert not bool(mask.all())


def test_token_expiration_forces_selected_even_under_zero_budget():
    l=layout()
    x=torch.zeros(1,13,4)
    cfg=GateConfig(ratios={},min_per_stream=0,
                   max_age=1,force_last_step_full=False)
    mask=TokenGate(cfg,10).select(x,x,torch.zeros(13,dtype=torch.long),
                                  l,chunk=0,step=2)
    assert int(mask.sum())==13
