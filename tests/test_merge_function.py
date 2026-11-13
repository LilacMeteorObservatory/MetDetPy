"""MergeFunction 的等价性回归测试。

`MergeFunction.max` 由 `np.max(stack, axis=0)` 改为逐帧原地 `np.maximum`：
结果必须逐元素一致，且不能改写调用方传入的帧。
"""
import sys

sys.path.insert(0, '.')

import numpy as np
import pytest

from MetLib.utils import MergeFunction


class TestMergeMax:
    @pytest.mark.parametrize("shape", [(8, 6, 3), (8, 6), (4, 4, 1)])
    @pytest.mark.parametrize("n", [1, 2, 5])
    def test_matches_numpy_reference(self, shape, n):
        rng = np.random.default_rng(0)
        frames = [rng.integers(0, 256, shape).astype(np.uint8) for _ in range(n)]
        expected = np.max(np.stack(frames, axis=0), axis=0)
        merged = MergeFunction.max(frames)
        assert np.array_equal(merged, expected)
        assert merged.dtype == expected.dtype

    def test_does_not_alias_caller_frames(self):
        frames = [np.zeros((4, 4, 3), dtype=np.uint8),
                  np.full((4, 4, 3), 9, dtype=np.uint8)]
        merged = MergeFunction.max(frames)
        frames[1][...] = 200
        assert merged.max() == 9

    def test_single_frame_is_copied(self):
        frame = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
        merged = MergeFunction.max([frame])
        assert np.array_equal(merged, frame)
        assert merged is not frame

    def test_mixed_dtype_falls_back_to_promotion(self):
        merged = MergeFunction.max([np.zeros((2, 2, 3), dtype=np.uint8),
                                    np.full((2, 2, 3), 1.5, dtype=np.float32)])
        assert merged.dtype == np.float32
        assert merged.max() == pytest.approx(1.5)

    def test_empty_stack_raises(self):
        with pytest.raises(ValueError):
            MergeFunction.max([])

    def test_accepts_ndarray_stack(self):
        rng = np.random.default_rng(1)
        stack = rng.integers(0, 256, (3, 5, 7, 3)).astype(np.uint8)
        assert np.array_equal(MergeFunction.max(stack),
                              np.max(stack, axis=0))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
