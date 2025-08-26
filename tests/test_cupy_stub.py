from importlib import reload
import sys
from pathlib import Path
import types
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ui_pages import cupy_stub as cp


def test_cupy_stub_exposes_array_helpers():
    data = [1, 2, 3]
    assert cp.array(data) is data
    assert cp.asarray(data) is data
    assert cp.asnumpy(data) is data


def test_model_builder_detects_stub():
    try:
        import sklearn  # type: ignore  # noqa: F401
    except Exception:
        pytest.skip("scikit-learn not installed")

    import training_pipeline.model_builder as mb

    cupy_module = types.ModuleType("cupy")
    cupy_module.__name__ = "ui_pages.cupy_stub"
    sys.modules["cupy"] = cupy_module
    mb = reload(mb)
    assert mb.CUPY_AVAILABLE is False
    sys.modules.pop("cupy", None)
