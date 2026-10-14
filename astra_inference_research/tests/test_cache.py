"""Synthetic CPU tests; NOT validation of an Astra checkpoint or CUDA performance."""
import pytest

torch = pytest.importorskip("torch")
from astra_research.cache import ResidualCache
from astra_research.instrument import AstraInstrumentation


def test_residual_reuse_and_refresh():
    cache = ResidualCache(enabled=True, indices=[0], interval=2, max_age=1)
    calls = []

    def compute(x, delta):
        calls.append(1)
        return x + delta

    x = torch.ones(1, 4)
    assert torch.equal(cache.execute(0, x, lambda: compute(x, 2),
                                     chunk=0, step=0, branch=0), x + 2)
    assert torch.equal(cache.execute(0, x + 1, lambda: compute(x, 99),
                                     chunk=0, step=1, branch=0), x + 3)
    assert len(calls) == 1
    cache.execute(0, x, lambda: compute(x, 3), chunk=0, step=2, branch=0)
    assert len(calls) == 2


def test_no_cross_branch_or_chunk_reuse():
    cache = ResidualCache(enabled=True, indices=[0], interval=2, max_age=1)
    x = torch.ones(2, 3)
    executed = []

    def full(tag):
        executed.append(tag)
        return x + 4

    cache.execute(0, x, lambda: full("b0"), chunk=0, step=0, branch=0)
    cache.execute(0, x, lambda: full("b1"), chunk=0, step=1, branch=1)
    cache.execute(0, x, lambda: full("new_chunk"), chunk=1, step=1, branch=0)
    assert executed == ["b0", "b1", "new_chunk"]
    assert cache.reused == 0


def test_cache_off_exact_forward():
    cache = ResidualCache(enabled=False, indices=[0], interval=2)
    x = torch.arange(8).float()
    calls = []
    for step in range(3):
        actual = cache.execute(0, x, lambda s=step: (calls.append(s), x + s)[1],
                               chunk=0, step=step, branch=0)
        assert torch.equal(actual, x + step)
    assert calls == [0, 1, 2]


class FakeBlock(torch.nn.Module):
    def forward(self, x):
        return x + 2


class FakeDiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([FakeBlock()])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class FakePipe:
    def __init__(self):
        self.dit = FakeDiT()
        self.scheduler = type("Scheduler", (), {"timesteps": list(range(50))})()


def test_instrumented_call_mapping():
    cfg = {
        "cache": {"method": "residual", "block_indices": [0], "interval": 2, "max_age": 1},
        "probe": {"enabled": False},
        "generation": {"use_camera_cfg": False, "text_guidance_scale": 1.0},
        "measurement": {"expected_denoising_steps": 50},
    }
    pipe = FakePipe()
    instrument = AstraInstrumentation(cfg, cache_enabled=True)
    instrument.install(pipe)
    x = torch.zeros(1, 3)
    for _ in range(100):
        result = pipe.dit(x)
        assert torch.allclose(result, x + 2)
    data = instrument.results()
    assert data["dit_forward_calls"] == 100
    assert data["complete_chunks_by_call_count"] == 2
    assert data["cache"]["reused_block_calls"] > 0
    assert instrument.current.chunk == 1


def test_threshold_rejection():
    cache = ResidualCache(enabled=True, indices=[0], relative_threshold=0.01)
    cache.execute(0, torch.zeros(2, 4), lambda: torch.ones(2, 4),
                  chunk=0, step=0, branch=0)
    cache.execute(0, torch.ones(2, 4) * 10, lambda: torch.ones(2, 4),
                  chunk=0, step=1, branch=0)
    assert cache.threshold_rejected == 1
