import numpy as np
import pytest

from MetLib.utils import SlidingWindow


@pytest.mark.parametrize("n", [1, 2, 3, 10, 30])
@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.float32])
def test_max_matches_ring_reduction(n, dtype):
    rng = np.random.default_rng(42)
    sw = SlidingWindow(n, (4, 5), dtype=dtype)
    for _ in range(5 * n + 7):
        frame = rng.integers(-20 if dtype != np.uint8 else 0, 100,
                             size=(4, 5)).astype(dtype)
        sw.update(frame)
        np.testing.assert_array_equal(sw.max, sw.sliding_window.max(axis=0))
        np.testing.assert_allclose(sw.mean,
                                   sw.sliding_window.sum(axis=0) / sw.length)
        frame.fill(0)  # The window must not retain the caller's mutable frame.


def test_negative_warmup_and_expiration():
    sw = SlidingWindow(3, (1,), dtype=np.int16)
    for value, expected in [(-1, 0), (-2, 0), (-3, -1), (-4, -2),
                            (-5, -3), (-6, -4), (-7, -5)]:
        sw.update(np.array([value], dtype=np.int16))
        assert sw.max[0] == expected


@pytest.mark.parametrize("count", [2, 5, 8])
def test_refresh_after_manual_edit(count):
    sw = SlidingWindow(5, (2, 3), dtype=np.uint8, force_int=True)
    for i in range(count):
        sw.update(np.full((2, 3), i, dtype=np.uint8))
    sw.sliding_window[0].fill(200)
    np.testing.assert_array_equal(sw.refresh_max(),
                                   sw.sliding_window.max(axis=0))
    # Manual edits to the raw window require independently refreshing the sum.
    sw.sum[:] = sw.sliding_window.sum(axis=0)
    for i in range(15):
        sw.update(np.full((2, 3), i, dtype=np.uint8))
        np.testing.assert_array_equal(sw.max, sw.sliding_window.max(axis=0))


def test_disabled_max_preserves_mean_and_std():
    sw = SlidingWindow(3, (2, 2), dtype=np.uint8, calc_std=True,
                       calc_max=False)
    assert sw.stack_max_cache is None
    sw.refresh_max = lambda: pytest.fail("Disabled max must not be refreshed")
    for i in range(8):
        sw.update(np.full((2, 2), i, dtype=np.uint8))
        values = sw.sliding_window[:sw.length]
        np.testing.assert_allclose(sw.mean, values.mean(axis=0))
        np.testing.assert_allclose(sw.std, values.std())
    with pytest.raises(RuntimeError, match="disabled"):
        _ = sw.max
    with pytest.raises(RuntimeError, match="disabled"):
        SlidingWindow.refresh_max(sw)


def test_invalid_window_length():
    with pytest.raises(ValueError, match="n must"):
        SlidingWindow(0, (2, 2))


@pytest.mark.parametrize("force_int", [False, True])
def test_mean_into_reuses_caller_buffer_without_changing_mean_snapshots(force_int):
    sw = SlidingWindow(2, (1,), dtype=np.uint8, force_int=force_int)
    sw.update(np.array([1], dtype=np.uint8))
    snapshot = sw.mean
    out = np.empty_like(snapshot)
    assert sw.mean_into(out) is out
    np.testing.assert_array_equal(out, [1])
    sw.update(np.array([2], dtype=np.uint8))
    assert sw.mean_into(out) is out
    np.testing.assert_array_equal(snapshot, [1])
    np.testing.assert_array_equal(out, [1] if force_int else [1.5])
    assert sw.mean is not snapshot
