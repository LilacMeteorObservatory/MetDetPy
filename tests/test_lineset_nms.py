import numpy as np
import pytest

from numpy.typing import NDArray
from MetLib.utils import lineset_nms, pt_len_sqr


def _reference_lineset_nms(
        lines: NDArray[np.int_]) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """
    Conduct NMS for line set.
    对线段合集执行NMS，并从线段集合中区分出“面积”类型。
    Args:
        lines (np.ndarray): 线段集合

    Returns:
        np.ndarray: 去重后的线段集合
    """
    # 其实也不需要阈值...每个输出都可以是百分之多少概率的直线。这会让整个概率体系更加可靠。
    # 合并线段的方法：线段的合并区域可以由其半径决定。
    # 合并概率的计算：根据合并后的长宽比
    num_line = len(lines)
    length_sqr = np.power((lines[:, 3] - lines[:, 1]), 2) + np.power(
        (lines[:, 2] - lines[:, 0]), 2)
    length_params = np.array([
        lines[:, 3] - lines[:, 1], lines[:, 0] - lines[:, 2],
        lines[:, 2] * lines[:, 1] - lines[:, 3] * lines[:, 0]
    ]).transpose()
    centers = (lines[:, 2:] + lines[:, :2]) // 2
    nms_ids = []
    nms_mask = np.zeros((num_line, ), dtype=np.uint8)
    length_sort = np.argsort(length_sqr)[::-1]
    # width_list 用于记录每个集合的宽度，在输出时给出直线比率。
    # TODO: 该机制如何与现有的流星机制结合也是一个问题。
    width_list = []
    # NMS
    for i, idx in enumerate(length_sort):
        # 如果已经被其他收纳 则忽略
        if nms_mask[idx]: continue
        # 开始新的一组
        nms_ids.append(idx)
        nms_mask[idx] = 1
        max_width = 0
        for idy in length_sort[i:]:
            if nms_mask[idy]: continue
            # 距离小于长线的length_sqr//4 (长线的半径以内) 即收纳.
            # TODO: 这个逻辑和过去并不一样。需要测试以验证稳定性。
            if pt_len_sqr(centers[idx], centers[idy]) < length_sqr[idx] // 4:
                nms_mask[idy] = 1
                # max_width only include Ax+By.
                max_width = max(
                    max_width,
                    np.abs(
                        np.sum(length_params[idx, :2] * centers[idy]) +
                        length_params[idx, -1]))
        width_list.append(max_width)

    # 后处理
    nms_lines = lines[nms_ids]
    # nonline_prob = |(Ax+By)+C|/sqrt(A^2+B^2) / LENGTH
    # *2 to calculate radius.
    nonline_prob = np.abs(width_list) / np.sqrt(
        np.sum(np.power(length_params[nms_ids, :2], 2), axis=1)) / np.sqrt(
            length_sqr[nms_ids]) * 2
    nonline_prob[nonline_prob > 1] = 1

    return nms_lines, nonline_prob



@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize("count", [1, 5, 30, 100, 500])
@pytest.mark.parametrize("short", [False, True])
def test_vectorized_nms_matches_reference(dtype, count, short):
    rng = np.random.default_rng(42 + count)
    starts = rng.integers([0, 0], [900, 480], size=(count, 2)).astype(dtype)
    ends = (starts + rng.integers(1, 40, size=(count, 2)).astype(dtype)
            if short else rng.integers([0, 0], [960, 540],
                                      size=(count, 2)).astype(dtype))
    lines = np.column_stack([starts, ends])
    original = lines.copy()
    expected = _reference_lineset_nms(lines)
    actual = lineset_nms(lines)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_array_equal(lines, original)


@pytest.mark.parametrize("lines", [
    [[0, 0, 20, 0], [0, 0, 20, 0], [20, 0, 0, 0]],
    [[0, 0, 20, 0], [10, 0, 30, 0], [9, 0, 29, 0]],
    [[0, 0, 20, 0], [0, 4, 20, 4], [0, 8, 20, 8]],
    [[0, 0, 0, 20], [0, 0, 20, 0], [20, 20, 0, 0]],
    [[0, 0, 0, 0], [5, 5, 5, 5]],
])
def test_nms_ties_boundary_directions_and_degenerate_lines(lines):
    lines = np.asarray(lines, dtype=np.int32)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = _reference_lineset_nms(lines)
        actual = lineset_nms(lines)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
