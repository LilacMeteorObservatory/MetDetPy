"""Unit tests for the NoiseSW noise estimator used by LineDetector/M3Detector.

The estimator replaced the "recompute ``np.std(window - mean)`` every
``interval*n`` frames" scheme with incremental first/second-moment accumulators
over a decimated sub-area, sampled with a stride and smoothed by a
time-constant-based EMA.  These tests pin the numerics (exactness of the
accumulators against brute force), the cold-start gate, the decimation
tolerance, the stride/time-constant bookkeeping and the LineDetector wiring.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from MetLib.Detector import LineDetector, NoiseSW
from MetLib.metstruct import (BinaryCoreCfg, BinaryCfg, DynamicCfg, HoughLineCfg,
                              MainDetectCfg)
from MetLib.utils import get_num_class


def make_mask(h=48, w=64):
    return np.ones((h, w), dtype=np.uint8)


def brute_force_sigma(window, decimate=1):
    """Reference: pooled per-pixel temporal variance of the (decimated) window."""
    win = window.astype(np.float64)
    if decimate > 1:
        win = win[:, ::decimate, ::decimate]
    return float(np.sqrt(np.mean(win.var(axis=0))))


def test_incremental_accumulators_match_bruteforce_over_sliding_window():
    rng = np.random.default_rng(7)
    n, h, w = 5, 32, 40
    frames = rng.integers(0, 255, (30, h, w), dtype=np.uint8)
    sw = NoiseSW(n, make_mask(h, w), fps=10.0, est_area=0.0, noise_interval=1,
                noise_decimate=1, noise_window_sec=1.0, noise_warmup=1)

    for i, frame in enumerate(frames):
        sw.update(frame)
        if sw.timer < n:
            assert sw.noise_sigma == 0.0
            continue
        expected = brute_force_sigma(sw.sliding_window)
        # The EMA is seeded by the first sample and then only updates once per
        # frame; momentum 0.9 on a static window keeps the value exact only at
        # the seed, so compare the raw measurement instead.
        assert sw.noise_sample == pytest.approx(expected, rel=1e-12), i


def test_accumulators_do_not_drift_on_a_stationary_window():
    rng = np.random.default_rng(11)
    n = 4
    frames = rng.integers(0, 255, (n, 32, 40), dtype=np.uint8)
    sw = NoiseSW(n, make_mask(32, 40), fps=10.0, est_area=0.0, noise_interval=1,
                noise_decimate=2, noise_window_sec=1.0, noise_warmup=10)
    values = []
    for _ in range(40):
        for frame in frames:
            sw.update(frame)
            if sw.timer >= n:
                values.append(sw.noise_sigma)
    tail = np.array(values[n:])
    assert np.allclose(tail, tail[0], rtol=1e-12)
    expected = brute_force_sigma(np.stack(frames), decimate=2)
    assert tail[0] == pytest.approx(expected, rel=1e-12)


def test_estimate_equals_pooled_variance_of_float_mean():
    rng = np.random.default_rng(3)
    n = 5
    window = rng.integers(0, 255, (n, 24, 32), dtype=np.uint8)
    sw = NoiseSW(n, make_mask(24, 32), fps=10.0, est_area=0.0, noise_interval=1,
                noise_decimate=1, noise_window_sec=1.0, noise_warmup=1)
    for frame in window:
        sw.update(frame)
    assert sw.noise_sigma == pytest.approx(brute_force_sigma(window), rel=1e-12)
    # ddof=0 (deviation from the *window* mean) is deliberate: the detector
    # thresholds ``max - mean``, so the residual scale is the relevant one.
    unbiased = float(np.sqrt(np.mean(window.astype(np.float64).var(axis=0,
                                                                  ddof=1))))
    assert sw.noise_sigma < unbiased


def test_cold_start_publishes_nothing_until_the_window_is_full():
    rng = np.random.default_rng(5)
    n = 6
    frames = rng.integers(0, 255, (n + 3, 24, 32), dtype=np.uint8)
    sw = NoiseSW(n, make_mask(24, 32), fps=8.0, est_area=0.0, noise_interval=1,
                noise_decimate=2, noise_window_sec=60.0, noise_warmup=5)
    published = []
    for frame in frames:
        sw.update(frame)
        published.append(sw.noise_sigma)
    assert published[:n - 1] == [0.0] * (n - 1)
    assert published[n - 1] > 0.0
    assert sw.noise_sample_count == 4


def test_decimation_keeps_the_estimate_close():
    rng = np.random.default_rng(13)
    n = 5
    frames = rng.integers(0, 255, (n, 64, 96), dtype=np.uint8)
    full = NoiseSW(n, make_mask(64, 96), fps=10.0, est_area=0.0, noise_interval=1,
                  noise_decimate=1, noise_window_sec=1.0, noise_warmup=1)
    dec = NoiseSW(n, make_mask(64, 96), fps=10.0, est_area=0.0, noise_interval=1,
                 noise_decimate=2, noise_window_sec=1.0, noise_warmup=1)
    for frame in frames:
        full.update(frame)
        dec.update(frame)
    assert dec.roi_shape == (32, 48)
    assert dec.noise_sigma == pytest.approx(full.noise_sigma, rel=0.03)


@pytest.mark.parametrize("fps,interval,window_sec", [
    (5.5556, 2, 60.0),
    (8.333, 1, 60.0),
    (25.0, 4, 30.0),
])
def test_ema_momentum_follows_the_time_constant(fps, interval, window_sec):
    sw = NoiseSW(4, make_mask(), fps=fps, est_area=0.0, noise_interval=interval,
                noise_window_sec=window_sec)
    period = interval / fps
    assert sw.noise_ema.init_momentum == pytest.approx(1.0 - period / window_sec)
    # one time constant == window_sec seconds of samples
    assert period / (1.0 - sw.noise_ema.init_momentum) == pytest.approx(
        window_sec)


def test_warmup_seeds_the_first_sample_and_ramps_the_momentum():
    rng = np.random.default_rng(17)
    n, warmup = 3, 4
    frames = rng.integers(0, 255, (n + warmup + 2, 24, 32), dtype=np.uint8)
    sw = NoiseSW(n, make_mask(24, 32), fps=10.0, est_area=0.0, noise_interval=1,
                noise_decimate=2, noise_window_sec=10.0, noise_warmup=warmup)
    m0 = sw.noise_ema.init_momentum
    moments = []
    for frame in frames:
        sw.update(frame)
        moments.append(sw.noise_ema.cur_momentum)
    ramp = moments[n - 1:n - 1 + warmup + 1]
    assert ramp[0] == pytest.approx(0.0)  # first sample seeds the EMA
    expected = [m0 * (1 - (1 - k / warmup)**2) for k in range(1, warmup)]
    assert ramp[1:warmup] == pytest.approx(expected)
    assert ramp[warmup] == pytest.approx(m0)  # warmup finished
    assert moments[-1] == pytest.approx(m0)


def test_dense_sampling_shrugs_off_a_single_contaminated_frame():
    rng = np.random.default_rng(23)
    n = 4
    stream = rng.integers(40, 60, (240, 32, 40), dtype=np.uint8)
    spiked = stream.copy()
    spiked[120, 8:16, 8:16] = 255  # a bright object crossing the sub-area once

    def run(frames):
        sw = NoiseSW(n, make_mask(32, 40), fps=10.0, est_area=0.0,
                    noise_interval=1, noise_decimate=2, noise_window_sec=60.0,
                    noise_warmup=10)
        values = []
        for frame in frames:
            sw.update(frame)
            values.append(sw.noise_sigma)
        return np.array(values)

    clean = run(stream)
    contaminated = run(spiked)
    # One contaminated frame stays inside the 4-frame window for 4 samples of
    # ~600 per time constant, so the smoothed level must barely move.
    live = clean > 0  # the pre-window frames publish nothing
    deviation = np.abs(contaminated[live] - clean[live]) / clean[live]
    assert deviation.max() < 0.05


def test_est_noise_disabled_updates_without_noise_tracking():
    sw = NoiseSW(3, make_mask(), est_noise=False)
    for _ in range(5):
        sw.update(np.zeros((48, 64), dtype=np.uint8))
    assert sw.timer == 5
    with pytest.raises(AssertionError):
        _ = sw.noise_sigma


def test_whole_frame_subarea_when_area_is_zero():
    sw = NoiseSW(3, make_mask(20, 30), fps=10.0, est_area=0.0, noise_decimate=2)
    assert sw.std_roi == (0, 0, 20, 30)
    assert sw.roi_shape == (10, 15)


def test_subarea_stays_inside_the_frame_and_follows_the_area_ratio():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, :] = 1  # only a band is valid sky
    sw = NoiseSW(3, mask, fps=10.0, est_area=0.1, noise_decimate=1)
    y0, x0, y1, x1 = sw.std_roi
    assert 0 <= y0 < y1 <= 100 and 0 <= x0 < x1 <= 100
    assert (y1 - y0) * (x1 - x0) == pytest.approx(1000, rel=0.2)


def make_binary_cfg():
    return BinaryCfg(binary=BinaryCoreCfg(adaptive_bi_thre=True,
                                          init_value=7,
                                          sensitivity="normal",
                                          area=0.1,
                                          interval=2),
                     hough_line=HoughLineCfg(threshold=10, min_len=10,
                                             max_gap=10),
                     dynamic=DynamicCfg(dy_mask=False, window_sec=5))


def test_line_detector_uses_fixed_threshold_until_the_estimate_exists():
    fps = 8.0
    cfg = make_binary_cfg()
    det = LineDetector(window_sec=1.0, fps=fps, mask=make_mask(64, 96),
                       cfg=cfg, logger=MagicMock())
    rng = np.random.default_rng(29)
    frames = rng.integers(0, 60, (12, 64, 96), dtype=np.uint8)
    thresholds = []
    for frame in frames:
        det.update(frame)
        thresholds.append(det.bi_threshold)
    n = det.stack.n
    assert thresholds[:n - 1] == [det.abs_sensitivity["normal"]] * (n - 1)
    # afterwards the quadratic mapping takes over
    sigma = det.stack.noise_sigma
    assert sigma > 0
    assert thresholds[-1] == round(1.2 * sigma**2 + 3.6)


def test_default_config_still_builds_an_m3_detector():
    from MetLib.Detector import M3Detector

    cfg = MainDetectCfg.from_json_file("config/m3det_normal.json")
    det = M3Detector(1, 8, make_mask(), cfg.detector.cfg, MagicMock())
    assert det.stack.n == 8
    assert det.stack.noise_sample_count == 0
    for _ in range(9):
        det.update(np.zeros((48, 64), dtype=np.uint8))
    assert det.stack.noise_sample_count > 0


def test_detectors_derive_num_cls_from_the_global_class_table():
    """类别列按全局类别 id 索引，类别数不接受调用方指定。"""
    from MetLib.Detector import BrightnessDetector, M3Detector
    from MetLib.metstruct import BrightnessCoreCfg, BrightnessCfg

    cfg = make_binary_cfg()
    expected = get_num_class()
    assert LineDetector(window_sec=1.0, fps=8.0, mask=make_mask(48, 64),
                        cfg=cfg, logger=MagicMock()).num_cls == expected
    assert M3Detector(1.0, 8.0, make_mask(48, 64), cfg,
                      MagicMock()).num_cls == expected
    assert BrightnessDetector(
        window_sec=1.0, fps=8.0, mask=make_mask(48, 64),
        cfg=BrightnessCfg(brightness=BrightnessCoreCfg(
            grid_rows=4, grid_cols=4, ema_momentum=0.9, z_threshold=3.0,
            abs_threshold=10.0, global_ratio=0.1, min_cell_valid=0.5)),
        logger=MagicMock()).num_cls == expected
    # 参数已移除：再显式传递应当是 TypeError，而不是被静默忽略。
    with pytest.raises(TypeError):
        LineDetector(window_sec=1.0, fps=8.0, mask=make_mask(48, 64),
                     num_cls=expected, cfg=cfg, logger=MagicMock())
