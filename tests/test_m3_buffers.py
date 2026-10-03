from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from MetLib.Detector import M3Detector
from MetLib.metstruct import MainDetectCfg
from MetLib.utils import SlidingWindow


@pytest.mark.parametrize("window_size", [3, 10])
def test_m3_reused_buffers_match_original_background_and_dynamic_mask(window_size):
    cfg = MainDetectCfg.from_json_file("config/m3det_normal.json")
    mask = np.ones((48, 64), dtype=np.uint8)
    detector = M3Detector(1, window_size, mask, 4, cfg.detector.cfg, MagicMock())
    history = SlidingWindow(window_size, mask.shape, dtype=np.uint8,
                            force_int=True, calc_max=False)
    rng = np.random.default_rng(42)
    mean_buffer = detector._mean_buffer
    diff_buffer = detector._diff_buffer
    mask_buffer = detector._dy_mask_buffer
    previous_dst = None
    previous_snapshot = None

    for frame in rng.integers(0, 50, (window_size + 5, *mask.shape), dtype=np.uint8):
        detector.update(frame)
        expected_mean = detector.stack.mean
        expected_diff = detector.stack.max - expected_mean
        median = cv2.medianBlur(expected_diff, 3)
        _, active = cv2.threshold(median, detector.bi_threshold, 255,
                                  cv2.THRESH_BINARY)
        active = cv2.morphologyEx(active, cv2.MORPH_CLOSE, detector.cv_op)
        history.update(active)
        dynamic_mask = np.array(
            history.sum <= (history.length - 1) * 255, dtype=np.uint8)
        dynamic_mask = cv2.erode(dynamic_mask, detector.cv_op)
        expected_dst = np.multiply(active, dynamic_mask)

        detector.detect()
        assert detector._mean_buffer is mean_buffer
        assert detector._diff_buffer is diff_buffer
        assert detector._dy_mask_buffer is mask_buffer
        np.testing.assert_array_equal(mean_buffer, expected_mean)
        np.testing.assert_array_equal(detector._diff_buffer, expected_diff)
        np.testing.assert_array_equal(detector.dst, expected_dst)
        if previous_dst is not None:
            np.testing.assert_array_equal(previous_dst, previous_snapshot)
        previous_dst = detector.dst
        previous_snapshot = detector.dst.copy()
