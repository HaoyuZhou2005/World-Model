"""CPU unit tests of 101501 scheduling logic, not a checkpoint/GPU test."""
import pytest

torch = pytest.importorskip("torch")

from astra_research.scheduled_cache import ScheduledResidualCache


def cache(**overrides):
    kwargs = dict(enabled=True, indices=[0, 1],
                  period_by_block={1: 1},
                  default_period=2, warmup_steps=1, max_age=1)
    kwargs.update(overrides)
    return ScheduledResidualCache(**kwargs)


def test_interval_two_reproduces_prior_refresh_positions():
    c = cache()
    x = torch.ones((1, 4))
    calls = []
    for s in range(6):
        y = c.execute(0, x, lambda s=s: (calls.append(s), x + s + 2)[1],
                      chunk=0, step=s, branch=0)
        assert torch.allclose(y, x + (s if s % 2 == 0 else s - 1) + 2)
    assert calls == [0, 2, 4]
    assert c.stats()["reused_block_calls"] == 3


def test_warmup_k_four_fully_computes_first_four_steps():
    c = cache(warmup_steps=4)
    x = torch.zeros((1, 2))
    calls = []
    for step in range(6):
        c.execute(0, x, lambda step=step: (calls.append(step), x + step)[1],
                  chunk=0, step=step, branch=0)
    assert calls == [0, 1, 2, 3, 4]
    assert c.reused == 1


def test_protected_layer_full_each_step():
    c = cache()
    x = torch.zeros((1, 3))
    called = []
    for step in range(6):
        c.execute(1, x, lambda step=step: (called.append(step), x + 1)[1],
                  chunk=0, step=step, branch=0)
    assert called == list(range(6))
    assert c.reused_by_block.get(1, 0) == 0


def test_state_reset_on_chunk_and_branch_separation():
    c = cache()
    x = torch.ones((1, 4))
    called = []
    run = lambda name: (called.append(name), x + 3)[1]
    c.execute(0, x, lambda: run("c0b0"), chunk=0, step=0, branch=0)
    c.execute(0, x, lambda: run("c0b1"), chunk=0, step=1, branch=1)
    c.execute(0, x, lambda: run("c1b0"), chunk=1, step=1, branch=0)
    assert called == ["c0b0", "c0b1", "c1b0"]
    assert c.reused == 0


def test_shape_guard_recomputes_on_mismatch():
    c = cache()
    x = torch.zeros(1, 4)
    c.execute(0, x, lambda: x + 2, chunk=0, step=0, branch=0)
    changed = torch.zeros(1, 5)
    called = []
    c.execute(0, changed, lambda: (called.append(1), changed + 3)[1],
              chunk=0, step=1, branch=0)
    assert len(called) == 1


def test_schedule_progressive_chunk_warmups():
    c = cache(warmup_by_chunk={"0": 1, "1": 2, "2": 4})
    assert c._must_refresh(0, 0, 1) is False
    assert c._must_refresh(0, 1, 1) is True
    assert c._must_refresh(0, 2, 3) is True
    assert c._must_refresh(0, 2, 5) is False


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError):
        cache(period_by_block={77: 2})
    with pytest.raises(ValueError):
        cache(warmup_steps=0)
    with pytest.raises(ValueError):
        cache(max_age=0)
    with pytest.raises(ValueError):
        cache(default_period=0)


def test_off_is_exact_forward():
    c = cache(enabled=False)
    x = torch.ones(2, 4)
    called = []
    for step in range(6):
        result = c.execute(0, x, lambda step=step: (called.append(step), x + step)[1],
                           chunk=0, step=step, branch=0)
        assert torch.equal(result, x + step)
    assert called == list(range(6))
