import pytest
torch=pytest.importorskip("torch")

from astra_token_research.collector import TeacherCollector
from astra_token_research.layout import PackedLayout
from astra_token_research.features import token_features,FEATURE_NAMES
from astra_token_research.future_safety import rollout_distillation_loss


def test_features_schema_and_teacher_save(tmp_path):
    layout=PackedLayout.from_shapes({"target":(2,1,10)},20)
    x=torch.ones(1,20,4)
    old=x.clone()
    previous_delta=torch.ones_like(x)
    ft=token_features(x,old,previous_delta,torch.zeros(20,dtype=torch.long),
                      layout,chunk=1,step=3,steps=50)
    assert ft.shape==(20,9)
    col=TeacherCollector(samples_per_step=5)
    col.observe(x,x+1,layout=layout,chunk=0,step=0,branch=0,block_id=0,steps=50)
    col.observe(x,x+2,layout=layout,chunk=0,step=1,branch=0,block_id=0,steps=50)
    destination=tmp_path/"teacher.pt"
    col.save(destination)
    saved=torch.load(destination,weights_only=True)
    assert saved["features"].shape==(5,9)
    assert tuple(saved["feature_names"])==FEATURE_NAMES
    assert torch.allclose(saved["oracle_local_error"],torch.ones(5))


def test_long_horizon_distillation_loss_is_separately_defined():
    teacher=[torch.zeros(1,2),torch.zeros(1,2)]
    student=[torch.ones(1,2),2*torch.ones(1,2)]
    loss=rollout_distillation_loss(student,teacher,distant_weight=1)
    assert loss.item()==pytest.approx(3.0)
    with pytest.raises(ValueError):
        rollout_distillation_loss(student,teacher[:1])
