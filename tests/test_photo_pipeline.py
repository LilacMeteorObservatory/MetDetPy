from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from MetLib.fileio import load_8bit_image, load_mask
from MetLib.model import (YOLOModel, _class_aware_nms, _cxcywh_to_tlwh,
                          _xyxy_to_tlwh)
from MetLib.utils import exclude_predictions_by_name


def _write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    path.write_bytes(encoded.tobytes())


@pytest.mark.parametrize(
    ("image", "case_name"),
    [
        (np.arange(12, dtype=np.uint8).reshape(3, 4), "grayscale"),
        (np.arange(48, dtype=np.uint8).reshape(3, 4, 4), "bgra"),
        (np.arange(36, dtype=np.uint16).reshape(3, 4, 3) * 1000, "uint16"),
    ],
)
def test_load_8bit_image_normalizes_source_layout_and_dtype(
        tmp_path: Path, image: np.ndarray, case_name: str):
    """All conventional inputs must satisfy the uint8 BGR contract."""
    image_path = tmp_path / f"{case_name}.png"
    _write_png(image_path, image)

    loaded = load_8bit_image(str(image_path))

    assert loaded.shape == (*image.shape[:2], 3)
    assert loaded.dtype == np.uint8


@pytest.mark.parametrize("shape", [(8, 8), (8, 8, 4)])
def test_yolo_forward_rejects_non_three_channel_input(shape: tuple[int, ...]):
    model = object.__new__(YOLOModel)
    model.c = 3

    with pytest.raises(AssertionError):
        model.forward(np.zeros(shape, dtype=np.uint8))


def test_yolo_forward_rejects_uint16_input():
    model = object.__new__(YOLOModel)
    model.c = 3

    with pytest.raises(AssertionError, match="uint8"):
        model.forward(np.full((2, 2, 3), 65535, dtype=np.uint16))


@pytest.mark.parametrize(("input_color_order", "expected"), [
    ("rgb", [30, 20, 10]),
    ("bgr", [10, 20, 30]),
])
def test_yolo_forward_converts_bgr_to_model_input_order(
        input_color_order: str, expected: list[int]):
    model = object.__new__(YOLOModel)
    model.c = 3
    model.dtype = np.float32
    model.multiscale_pred = 0
    model.input_color_order = input_color_order
    captured: dict[str, np.ndarray] = {}

    def fake_forward(image: np.ndarray):
        captured["image"] = image
        return np.array([[0, 0, 1, 1]], dtype=int), np.array([[0.25]],
                                                               dtype=float)

    model._forward = fake_forward
    bgr_pixel = np.array([[[10, 20, 30]]], dtype=np.uint8)
    original = bgr_pixel.copy()
    _, scores = model.forward(bgr_pixel)

    np.testing.assert_allclose(captured["image"][0, 0],
                               np.asarray(expected) / 255)
    np.testing.assert_array_equal(bgr_pixel, original)
    np.testing.assert_allclose(scores, [[0.5]])


def test_yolo_forward_rejects_unknown_model_input_color_order():
    model = object.__new__(YOLOModel)
    model.c = 3
    model.dtype = np.float32
    model.multiscale_pred = 0
    model.input_color_order = "auto"

    with pytest.raises(ValueError, match="input color order"):
        model.forward(np.zeros((1, 1, 3), dtype=np.uint8))


def test_exclude_predictions_keeps_boxes_and_probabilities_aligned():
    boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30],
                      [40, 40, 50, 50]])
    probabilities = np.zeros((3, 8), dtype=float)
    probabilities[0, 0] = 0.9  # METEOR
    probabilities[1, 1] = 0.8  # PLANE/SATELLITE
    probabilities[2, 7] = 0.7  # BUGS

    kept_boxes, kept_probabilities = exclude_predictions_by_name(
        boxes, probabilities, ["PLANE/SATELLITE", "BUGS"])

    np.testing.assert_array_equal(kept_boxes, boxes[[0]])
    np.testing.assert_array_equal(kept_probabilities, probabilities[[0]])


def test_load_mask_handles_grayscale_uint16(tmp_path: Path):
    mask_path = tmp_path / "gray16.png"
    _write_png(mask_path, np.array([[0, 65535]], dtype=np.uint16))

    mask = load_mask(str(mask_path), grayscale=True)

    np.testing.assert_array_equal(mask, np.array([[0, 1]], dtype=np.uint8))


def test_load_mask_handles_color_and_uses_fully_opaque_alpha(tmp_path: Path):
    color_path = tmp_path / "color.png"
    color = np.array([[[255, 255, 255], [0, 0, 0], [0, 0, 255]]],
                     dtype=np.uint8)
    _write_png(color_path, color)
    np.testing.assert_array_equal(
        load_mask(str(color_path), grayscale=True),
        np.array([[1, 0, 0]], dtype=np.uint8))

    bgra_path = tmp_path / "opaque.png"
    opaque_bgra = np.array([[[255, 255, 255, 255], [0, 0, 0, 255]]],
                           dtype=np.uint8)
    _write_png(bgra_path, opaque_bgra)
    np.testing.assert_array_equal(
        load_mask(str(bgra_path), grayscale=True),
        np.array([[0, 0]], dtype=np.uint8))


def test_load_mask_preserves_transparency_mask_semantics(tmp_path: Path):
    mask_path = tmp_path / "alpha.png"
    bgra = np.array([[[0, 0, 0, 0], [255, 255, 255, 255]]],
                    dtype=np.uint8)
    _write_png(mask_path, bgra)

    mask = load_mask(str(mask_path), grayscale=False)

    assert mask.shape == (1, 2, 3)
    np.testing.assert_array_equal(mask[0, :, 0],
                                  np.array([1, 0], dtype=np.uint8))
    np.testing.assert_array_equal(mask[:, :, 0], mask[:, :, 1])
    np.testing.assert_array_equal(mask[:, :, 1], mask[:, :, 2])


def _nms_indices(boxes: np.ndarray, scores: list[float]) -> list[int]:
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores, 0.01, 0.45)
    return np.asarray(indices).reshape(-1).tolist()


def test_yolo_center_boxes_convert_directly_to_opencv_tlwh():
    boxes_cxcywh = np.array([[50, 60, 20, 10]], dtype=float)

    np.testing.assert_array_equal(_cxcywh_to_tlwh(boxes_cxcywh),
                                  np.array([[40, 55, 20, 10]], dtype=float))


def test_passing_xyxy_to_opencv_nms_can_false_suppress_a_separate_box():
    boxes_xyxy = np.array([[1000, 100, 1100, 200],
                           [1080, 100, 1180, 200]], dtype=float)
    class_scores = np.array([[0.95], [0.90]])

    assert _nms_indices(boxes_xyxy, [0.95, 0.90]) == [0]
    boxes_tlwh = _xyxy_to_tlwh(boxes_xyxy)
    assert _nms_indices(boxes_tlwh, [0.95, 0.90]) == [0, 1]
    assert _class_aware_nms(boxes_tlwh, class_scores, 0.01,
                            0.45).tolist() == [0, 1]


def test_passing_xyxy_to_opencv_nms_can_miss_a_duplicate_box():
    boxes_xyxy = np.array([[0, 0, 100, 100],
                           [20, 20, 120, 120]], dtype=float)
    class_scores = np.array([[0.95], [0.90]])

    assert _nms_indices(boxes_xyxy, [0.95, 0.90]) == [0, 1]
    boxes_tlwh = _xyxy_to_tlwh(boxes_xyxy)
    assert _nms_indices(boxes_tlwh, [0.95, 0.90]) == [0]
    assert _class_aware_nms(boxes_tlwh, class_scores, 0.01,
                            0.45).tolist() == [0]


def test_objectness_only_nms_can_keep_the_worse_classified_candidate():
    boxes_xyxy = np.array([[100, 100, 200, 200],
                           [100, 100, 200, 200]], dtype=float)
    objectness = [0.95, 0.75]
    objectness_times_class = [0.95 * 0.05, 0.75 * 0.90]
    calibrated_scores = np.sqrt(
        np.asarray(objectness_times_class, dtype=float))[:, None]

    boxes_tlwh = _xyxy_to_tlwh(boxes_xyxy)
    assert _nms_indices(boxes_tlwh, objectness) == [0]
    assert _nms_indices(boxes_tlwh, objectness_times_class) == [1]
    assert _class_aware_nms(boxes_tlwh, calibrated_scores, 0.01,
                            0.45).tolist() == [1]


def test_class_aware_nms_keeps_coincident_different_classes():
    boxes_xyxy = np.array([[100, 100, 200, 200],
                           [100, 100, 200, 200]], dtype=float)
    class_scores = np.array([[0.90, 0.05], [0.05, 0.85]])

    assert _class_aware_nms(_xyxy_to_tlwh(boxes_xyxy), class_scores, 0.01,
                            0.45).tolist() == [0, 1]


def _model_with_raw_predictions(predictions: np.ndarray) -> YOLOModel:
    model = object.__new__(YOLOModel)
    model.c = 3
    model.h = 8
    model.w = 8
    model.resize = False
    model.scale_h = 1
    model.scale_w = 1
    model.unwarning = False
    model.nms = True
    model.input_color_order = "rgb"
    model.pos_thre = 0.25
    model.nms_thre = 0.45
    model.backend = SimpleNamespace(
        forward=lambda _image: [[predictions.copy()]])
    return model


def test_single_tile_candidate_gate_uses_joint_score():
    predictions = np.array([
        [2, 2, 2, 2, 0.20, 0.90],
        [6, 6, 2, 2, 0.90, 0.05],
    ], dtype=np.float32)
    model = _model_with_raw_predictions(predictions)
    model.pos_thre = 0.10

    boxes, joint_scores = model._forward(
        np.zeros((8, 8, 3), dtype=np.float32))

    # Candidate eligibility follows objectness * class probability rather than
    # objectness alone.
    assert boxes.tolist() == [[1, 1, 3, 3]]
    np.testing.assert_allclose(joint_scores, [[0.20 * 0.90]], rtol=1e-6)


def test_single_tile_nms_ranks_eligible_candidates_by_joint_score():
    predictions = np.array([
        [4, 4, 2, 2, 0.95, 0.05],
        [4, 4, 2, 2, 0.75, 0.90],
    ], dtype=np.float32)
    model = _model_with_raw_predictions(predictions)

    boxes, joint_scores = model._forward(
        np.zeros((8, 8, 3), dtype=np.float32))

    assert boxes.tolist() == [[3, 3, 5, 5]]
    np.testing.assert_allclose(joint_scores, [[0.75 * 0.90]], rtol=1e-6)


def test_multiscale_merge_uses_configured_nms_threshold():
    model = object.__new__(YOLOModel)
    model.c = 3
    model.dtype = np.float32
    model.multiscale_pred = 1
    model.multiscale_partition = 2
    model.hw_ratio = 1.0
    model.hw_tolerance = 0.2
    model.pos_thre = 0.25
    model.nms_thre = 0.45
    model.input_color_order = "rgb"
    model.logger = SimpleNamespace(debug=lambda *args: None,
                                   error=lambda *args: None)

    def fake_forward(_image: np.ndarray):
        boxes = np.array([[10, 10, 110, 110], [90, 10, 190, 110]])
        scores = np.array([[0.90], [0.80]])
        return boxes, scores

    model._forward = fake_forward
    boxes, _ = model.forward(np.zeros((200, 200, 3), dtype=np.uint8))

    assert boxes.tolist() == [[10, 10, 110, 110], [90, 10, 190, 110]]


def test_multiscale_merge_uses_joint_score_threshold_directly():
    model = object.__new__(YOLOModel)
    model.c = 3
    model.dtype = np.float32
    model.multiscale_pred = 1
    model.multiscale_partition = 2
    model.hw_ratio = 1.0
    model.hw_tolerance = 0.2
    model.pos_thre = 0.06
    model.nms_thre = 0.45
    model.input_color_order = "rgb"
    model.logger = SimpleNamespace(debug=lambda *args: None,
                                   error=lambda *args: None)

    def fake_forward(_image: np.ndarray):
        boxes = np.array([[10, 10, 30, 30], [60, 60, 80, 80]])
        joint_scores = np.array([[0.07], [0.05]])
        return boxes, joint_scores

    model._forward = fake_forward
    boxes, scores = model.forward(np.zeros((100, 100, 3), dtype=np.uint8))

    # Both NMS passes use the configured raw joint-score threshold. The final
    # public score is still sqrt(objectness * class probability).
    assert boxes.tolist() == [[10, 10, 30, 30]]
    np.testing.assert_allclose(scores, [[np.sqrt(0.07)]])
