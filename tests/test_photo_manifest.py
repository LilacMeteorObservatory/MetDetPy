import json
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

import MetDetPhoto
from MetLib.image_manifest import load_image_manifest
from MetLib.model import YOLOModel
from tools.compare_photo import compare_records, load_records


@pytest.fixture(autouse=True)
def fresh_logger(monkeypatch):
    from MetLib import metlog
    monkeypatch.setattr(metlog, 'met_logger', metlog.ThreadMetLog())


def make_manifest(tmp_path):
    paths = [tmp_path / 'a.png', tmp_path / 'b.png']
    for path in paths:
        ok, data = cv2.imencode('.png', np.zeros((8, 8, 3), np.uint8))
        assert ok
        path.write_bytes(data.tobytes())
    manifest = tmp_path / 'images.txt'
    manifest.write_text('a.png\n\nb.png\n', encoding='utf-8')
    return manifest, paths


def test_manifest_relative_order_blank_lines(tmp_path):
    manifest, paths = make_manifest(tmp_path)
    assert load_image_manifest(str(manifest)) == list(map(str, paths))


@pytest.mark.parametrize('text,error', [('', ValueError), ('missing.png', FileNotFoundError),
                                        ('images.txt', ValueError)])
def test_manifest_rejects_invalid_entries(tmp_path, text, error):
    manifest, _ = make_manifest(tmp_path)
    manifest.write_text(text, encoding='utf-8')
    with pytest.raises(error):
        load_image_manifest(str(manifest))


def test_manifest_deduplicates_resolved_paths_with_warning(tmp_path):
    manifest, paths = make_manifest(tmp_path)
    manifest.write_text(f'a.png\nb.png\n./a.png\n{paths[1]}\n', encoding='utf-8')
    with pytest.warns(UserWarning, match='duplicate image skipped') as warnings:
        assert load_image_manifest(str(manifest)) == list(map(str, paths))
    assert len(warnings) == 2


def test_cli_device_and_sparse_results(tmp_path, monkeypatch):
    manifest, paths = make_manifest(tmp_path)
    result = tmp_path / 'result.json'
    fake_model = Mock()
    fake_model.forward.side_effect = [(np.array([[1, 2, 3, 4]]), np.array([[0.91]*9])),
                                      (np.empty((0, 4)), np.empty((0, 9)))]
    constructor = Mock(return_value=fake_model)
    monkeypatch.setattr(MetDetPhoto, 'YOLOModel', constructor)
    MetDetPhoto.main(['@'+str(manifest), '--device', 'cpu', '--save-path', str(result)])
    assert constructor.call_args.kwargs['providers_key'] == 'cpu'
    data = json.loads(result.read_text(encoding='utf-8'))
    assert data['type'] == 'image-prediction'
    assert len(data['results']) == 1
    assert data['results'][0]['img_filename'] == str(paths[0])
    assert data['results'][0]['boxes'] == [[1, 2, 3, 4]]


def test_cli_load_failure_is_logged_and_skipped(tmp_path, monkeypatch):
    manifest, paths = make_manifest(tmp_path)
    paths[1].write_bytes(b'corrupt')
    fake_model = Mock()
    fake_model.forward.return_value = (np.empty((0, 4)), np.empty((0, 9)))
    monkeypatch.setattr(MetDetPhoto, 'YOLOModel', Mock(return_value=fake_model))
    logger = Mock()
    monkeypatch.setattr(MetDetPhoto, 'get_default_logger', lambda: logger)
    result = tmp_path / 'result.json'
    MetDetPhoto.main(['@'+str(manifest), '--save-path', str(result)])
    logger.error.assert_any_call(f'Failed to load image {paths[1]}.')
    assert fake_model.forward.call_count == 1
    assert json.loads(result.read_text(encoding='utf-8'))['results'] == []


def test_multiscale_failure_is_logged_and_returns_empty_results():
    model = object.__new__(YOLOModel)
    model.c, model.dtype, model.input_color_order = 3, np.float32, 'rgb'
    model.multiscale_pred, model.multiscale_partition = 2, 2
    model.hw_ratio, model.hw_tolerance = 1, 0.2
    model.logger = Mock()
    model._forward = Mock(side_effect=RuntimeError('inference failed'))
    boxes, scores = model.forward(np.zeros((8, 8, 3), np.uint8))
    assert len(boxes) == len(scores) == 0
    assert 'inference failed' in model.logger.error.call_args.args[0]


@pytest.mark.parametrize('scale', [0, 2])
def test_cli_backend_timeout_is_logged(tmp_path, monkeypatch, scale):
    from MetLib import model

    manifest, _ = make_manifest(tmp_path)
    backend = Mock()
    backend.input_shape = [[1, 3, 8, 8]]
    backend.forward.return_value = None
    monkeypatch.setitem(model.SUFFIX2BACKEND, 'onnx', Mock(return_value=backend))
    logger = Mock()
    monkeypatch.setattr(MetDetPhoto, 'get_default_logger', lambda: logger)
    result = tmp_path / 'result.json'
    MetDetPhoto.main(['@'+str(manifest), '--scale', str(scale),
                     '--save-path', str(result)])
    assert any('backend lock' in call.args[0] for call in logger.warning.call_args_list)
    assert json.loads(result.read_text(encoding='utf-8'))['results'] == []


def test_default_model_keeps_tolerant_timeout_behavior(monkeypatch):
    from MetLib import model

    backend = Mock()
    backend.input_shape = [[1, 3, 8, 8]]
    backend.forward.return_value = None
    monkeypatch.setitem(model.SUFFIX2BACKEND, 'onnx', Mock(return_value=backend))
    instance = YOLOModel('model.onnx', 'float32', logger=Mock())
    boxes, scores = instance.forward(np.zeros((8, 8, 3), np.uint8))
    assert len(boxes) == len(scores) == 0


def test_cli_inference_failure_is_logged(tmp_path, monkeypatch):
    manifest, _ = make_manifest(tmp_path)
    fake_model = Mock()
    fake_model.forward.side_effect = RuntimeError('inference failed')
    monkeypatch.setattr(MetDetPhoto, 'YOLOModel', Mock(return_value=fake_model))
    logger = Mock()
    monkeypatch.setattr(MetDetPhoto, 'get_default_logger', lambda: logger)
    result = tmp_path / 'result.json'
    MetDetPhoto.main(['@'+str(manifest), '--save-path', str(result)])
    assert 'inference failed' in logger.error.call_args.args[0]
    assert json.loads(result.read_text(encoding='utf-8'))['results'] == []


def test_cli_unavailable_device_is_rejected(monkeypatch):
    monkeypatch.setattr(MetDetPhoto, 'AVAILABLE_DEVICE_ALIAS', ['cpu'])
    with pytest.raises(SystemExit) as error:
        MetDetPhoto.main(['unused.png', '--device', 'cuda'])
    assert error.value.code == 2


def test_comparison_order_empty_class_score_and_unmatched():
    x = ([0, 0, 10, 10], 'METEOR', '0.90')
    y = ([20, 20, 30, 30], 'BUGS', '0.80')
    baseline = {'a': [x, y], 'b': [x], 'c': [x], 'e': [x]}
    candidate = {'a': [y, x], 'b': [([1, 0, 10, 10], 'BUGS', '0.91')],
                 'e': [(x[0], x[1], '0.89')]}
    report = compare_records(['a', 'b', 'c', 'd', 'e'], baseline, candidate)
    summary = report['summary']
    assert summary['exact_box_class_images'] == 3
    assert summary['exact_saved_result_images'] == 2
    assert summary['both_empty_images'] == 1
    assert summary['class_changed_pairs'] == 1
    assert summary['baseline_unmatched'] == 1
    assert summary['candidate_unmatched'] == 0
    assert summary['max_coordinate_delta'] == 1


def test_comparison_rejects_unknown_images(tmp_path):
    result = tmp_path / 'result.json'
    result.write_text(json.dumps({'type': 'image-prediction', 'results': [
        dict(img_filename='unknown.png', boxes=[], preds=[], prob=[])]}), encoding='utf-8')
    with pytest.raises(ValueError, match='Unknown or duplicate'):
        load_records(result, set())
