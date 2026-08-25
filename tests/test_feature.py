from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

import MetLib.Detector as detector_module
from MetLib.Detector import MLDetector
from MetLib.feature import calc_roi_gradient


@pytest.mark.parametrize("shape", [(0, 10), (10, 0, 3)])
def test_calc_roi_gradient_returns_nan_for_empty_image(shape):
    image = np.zeros(shape, dtype=np.uint8)

    assert np.isnan(calc_roi_gradient(image))


def test_calc_roi_gradient_returns_nan_without_gradient():
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    assert np.isnan(calc_roi_gradient(image))


def _make_ml_detector(positions, image=None):
    detector = object.__new__(MLDetector)
    detector.stack = SimpleNamespace(
        max=np.zeros((20, 20, 3), dtype=np.uint8)
        if image is None else image)
    detector.model = MagicMock()
    detector.model.forward.return_value = (
        np.asarray(positions, dtype=int),
        np.ones((len(positions), 1), dtype=np.float64),
    )
    detector.logger = MagicMock()
    return detector


def test_ml_detector_swaps_y_coordinates_for_finite_reverse_gradient(
        monkeypatch):
    detector = _make_ml_detector([[2, 3, 8, 9]])
    monkeypatch.setattr(detector_module, "calc_roi_gradient",
                        lambda _image: np.pi / 2)

    positions, _ = detector.detect()

    np.testing.assert_array_equal(positions, [[2, 9, 8, 3]])


def test_ml_detector_keeps_coordinates_for_non_finite_gradient(monkeypatch):
    detector = _make_ml_detector([[2, 3, 8, 9]])
    monkeypatch.setattr(detector_module, "calc_roi_gradient",
                        lambda _image: float("nan"))

    positions, _ = detector.detect()

    np.testing.assert_array_equal(positions, [[2, 3, 8, 9]])


def test_ml_detector_handles_out_of_bounds_empty_roi():
    detector = _make_ml_detector([[30, 30, 40, 40]])

    positions, _ = detector.detect()

    np.testing.assert_array_equal(positions, [[30, 30, 40, 40]])
