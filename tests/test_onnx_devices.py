from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from MetLib import model


@pytest.mark.parametrize("alias,provider", [
    ("dml", "DmlExecutionProvider"), ("cuda", "CUDAExecutionProvider")])
def test_indexed_provider_options(monkeypatch, alias, provider):
    session = Mock()
    session.get_inputs.return_value = [SimpleNamespace(shape=[1, 3, 2, 2], name="in0")]
    session.get_providers.return_value = [provider, "CPUExecutionProvider"]
    constructor = Mock(return_value=session)
    monkeypatch.setattr(model.ort, "InferenceSession", constructor)
    monkeypatch.setattr(model.ort, "get_available_providers", lambda: [provider])
    monkeypatch.setattr(model, "is_lfs_pointer", lambda _: False)
    backend = model.ONNXBackend("model.onnx", np.float32, False, f"{alias}:1")
    constructor.assert_called_once_with(
        "model.onnx", providers=[(provider, {"device_id": "1"}), "CPUExecutionProvider"],
        enable_fallback=False)
    assert backend.device == f"{provider}:1"


@pytest.mark.parametrize("key", ["dml:-1", "cuda:x", "dml:", "dml:1:2", "cpu:0", "default:1", "vulkan:1"])
def test_invalid_device_key(key):
    with pytest.raises(ValueError):
        model.validate_provider_key(key)


@pytest.mark.parametrize("key", ["default", "cpu", "dml", "cuda", "coreml", "dml:0", "cuda:1"])
def test_valid_device_key(key):
    assert model.validate_provider_key(key) == key


def test_missing_indexed_provider_does_not_fall_back(monkeypatch):
    monkeypatch.setattr(model.ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    with pytest.raises(ValueError, match="not installed"):
        model.ONNXBackend("model.onnx", np.float32, False, "dml:1")
