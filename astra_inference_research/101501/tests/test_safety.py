"""Safety parsing regression tests; no nvidia-smi or CUDA needed."""
import pytest
from astra_research.safety import selected_physical_gpu


@pytest.mark.parametrize("value", ["0,1", "", "all", None, "GPU-abcd", "-1"])
def test_reject_implicit_or_multiple_gpu_visibility(value):
    with pytest.raises(RuntimeError):
        selected_physical_gpu(value)


@pytest.mark.parametrize("value,result", [("0", 0), ("5", 5), (" 7 ", 7)])
def test_one_explicit_physical_gpu(value, result):
    assert selected_physical_gpu(value) == result
