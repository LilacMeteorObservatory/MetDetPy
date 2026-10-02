"""Thread configuration tests using mocked sessions, without inference or decoding."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from MetLib import model
from MetLib.metstruct import MainDetectCfg, ModelCfg


@pytest.fixture
def session_constructor(monkeypatch):
    session = Mock()
    session.get_inputs.return_value = [SimpleNamespace(shape=[1, 3, 2, 2], name="images")]
    session.get_providers.return_value = ["CPUExecutionProvider"]
    constructor = Mock(return_value=session)
    monkeypatch.setattr(model.ort, "InferenceSession", constructor)
    monkeypatch.setattr(model, "is_lfs_pointer", lambda _: False)
    return constructor


def legacy_model_config():
    config_path = Path(__file__).resolve().parents[1] / "config" / "dldet.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    for model_cfg in (cfg["detector"]["cfg"]["model"],
                      cfg["collector"]["recheck_cfg"]["model"]):
        model_cfg.pop("num_threads", None)
        model_cfg["warmup"] = False
    return cfg


def test_legacy_config_defaults_to_automatic_threads(session_constructor):
    cfg = MainDetectCfg.from_dict(legacy_model_config())
    main = cfg.detector.cfg.model
    recheck = cfg.collector.recheck_cfg.model
    assert main.num_threads == recheck.num_threads == 0
    model.init_model(main, logger=Mock())
    model.init_model(recheck, logger=Mock())
    assert session_constructor.call_count == 2
    for call in session_constructor.call_args_list:
        assert call.kwargs["sess_options"].intra_op_num_threads == 0


def test_main_and_recheck_thread_settings_reach_separate_sessions(session_constructor):
    data = legacy_model_config()
    data["detector"]["cfg"]["model"]["num_threads"] = 2
    data["collector"]["recheck_cfg"]["model"]["num_threads"] = 4
    cfg = MainDetectCfg.from_dict(data)
    model.init_model(cfg.detector.cfg.model, logger=Mock())
    model.init_model(cfg.collector.recheck_cfg.model, logger=Mock())
    options = [call.kwargs["sess_options"] for call in session_constructor.call_args_list]
    assert [option.intra_op_num_threads for option in options] == [2, 4]
    assert options[0] is not options[1]
    assert cfg.to_dict()["collector"]["recheck_cfg"]["model"]["num_threads"] == 4


@pytest.mark.parametrize("num_threads", [0, 1, 4])
def test_direct_backend_thread_setting(session_constructor, num_threads):
    model.ONNXBackend("model.onnx", np.float32, False, "cpu", num_threads=num_threads)
    assert session_constructor.call_args.kwargs["providers"] == ["CPUExecutionProvider"]
    options = session_constructor.call_args.kwargs["sess_options"]
    assert options.intra_op_num_threads == num_threads
    assert options.execution_mode == model.ort.ExecutionMode.ORT_SEQUENTIAL
    assert options.inter_op_num_threads == 0


@pytest.mark.parametrize("num_threads", [-1, 1.5, "4", True, None])
def test_direct_backend_rejects_invalid_threads_before_session_creation(
        session_constructor, num_threads):
    with pytest.raises(ValueError, match="num_threads"):
        model.ONNXBackend("model.onnx", np.float32, False, "cpu", num_threads=num_threads)
    session_constructor.assert_not_called()


@pytest.mark.parametrize("num_threads", [-1, 1.5, "4", True, None])
def test_model_config_rejects_invalid_threads(num_threads):
    kwargs = legacy_model_config()["detector"]["cfg"]["model"]
    with pytest.raises(ValueError, match="num_threads"):
        ModelCfg(**kwargs, num_threads=num_threads)


def test_json_config_rejects_boolean_thread_count():
    data = legacy_model_config()["detector"]["cfg"]["model"]
    data["num_threads"] = True
    with pytest.raises(ValueError, match="num_threads"):
        ModelCfg.from_dict(data)
