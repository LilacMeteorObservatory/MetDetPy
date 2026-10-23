"""Geometric invariants for aspect adaptation and complete image coverage."""
from types import SimpleNamespace

import numpy as np
import pytest

from MetLib.model import YOLOModel, _plan_prediction_tiles, _tile_starts


@pytest.mark.parametrize("length,size", [(720, 276), (720, 144), (576, 408),
                                         (7, 2), (1, 1), (3840, 1067)])
def test_axis_windows_cover_both_edges_without_gaps(length, size):
    starts = _tile_starts(length, size)
    assert starts[0] == 0 and starts[-1] + size == length
    assert all(a < b <= a + size for a, b in zip(starts, starts[1:]))
    assert all(0 <= start < start + size <= length for start in starts)


def test_pal_landscape_uses_two_native_ratio_crops_without_transposing():
    transpose, levels = _plan_prediction_tiles(576, 720, 544 / 960, .2, 2, 2)
    assert not transpose
    assert levels[0] == [(0, 0, 720, 408), (0, 168, 720, 576)]
    assert len(levels[1]) == 6
    assert _plan_prediction_tiles(576, 720, 544 / 960, .2, 1, 2)[1] == levels[:1]


@pytest.mark.parametrize("height,width,transposed", [
    (480, 720, False), (720, 480, True), (1080, 1920, False),
    (2160, 3840, False), (3072, 3840, False), (3840, 2160, True),
    (576, 740, False), (1000, 1000, False), (40, 4000, False),
])
def test_levels_keep_ratio_and_cover_every_axis(height, width, transposed):
    transpose, levels = _plan_prediction_tiles(height, width, 544 / 960, .2, 3, 2)
    assert transpose == transposed
    if transpose:
        height, width = width, height
    for windows in levels:
        assert min(x1 for x1, _, _, _ in windows) == 0
        assert min(y1 for _, y1, _, _ in windows) == 0
        assert max(x2 for _, _, x2, _ in windows) == width
        assert max(y2 for _, _, _, y2 in windows) == height
        assert all(0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
                   for x1, y1, x2, y2 in windows)
        xs = sorted({(x1, x2) for x1, _, x2, _ in windows})
        ys = sorted({(y1, y2) for _, y1, _, y2 in windows})
        assert all(b[0] <= a[1] for a, b in zip(xs, xs[1:]))
        assert all(b[0] <= a[1] for a, b in zip(ys, ys[1:]))
        assert len(windows) == len(xs) * len(ys)
        # Only integer rounding and the existing tolerance may stretch a crop.
        assert all(abs((y2-y1)/(x2-x1) - 544/960) < .21
                   for x1, y1, x2, y2 in windows)


def test_resolution_alone_does_not_turn_scale_one_into_multiscale():
    for height, width in [(540, 960), (1080, 1920), (2160, 3840)]:
        _, levels = _plan_prediction_tiles(height, width, 544 / 960, .2, 2, 2)
        assert levels[0] == [(0, 0, width, height)]
        assert len(levels[1]) == 4


@pytest.mark.parametrize("height,width", [(576, 720), (720, 480)])
def test_forward_maps_edge_boxes_back_to_original_coordinates(height, width):
    model = object.__new__(YOLOModel)
    model.c, model.dtype, model.input_color_order = 3, np.float32, "bgr"
    model.hw_ratio, model.hw_tolerance = 544/960, .2
    model.multiscale_pred, model.multiscale_partition = 1, 2
    model.pos_thre, model.nms_thre = .25, .99
    model.logger = SimpleNamespace(debug=lambda *_: None, error=lambda *_: None)
    model._forward = lambda image: (
        np.array([[0, 0, image.shape[1], image.shape[0]]]), np.array([[1.0]]))
    boxes, _ = model.forward(np.zeros((height, width, 3), np.uint8))
    assert boxes[:, 0].min() == boxes[:, 1].min() == 0
    assert boxes[:, 2].max() == width
    assert boxes[:, 3].max() == height
