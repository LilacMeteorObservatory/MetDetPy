"""叠帧累积与掩膜后置（mask hoisting）的等价性回归测试。

背景：4K 下复核叠帧曾是主要开销，两处实现被优化：
1. `MaxImgContainer.append` 由 `np.max([a, b], axis=0)` 改为 `np.maximum(a, b)`；
2. 掩膜由"逐帧乘法"改为"叠帧后乘一次"，依赖掩膜为 0/1 二值这一事实：

       max_i(m ⊙ x_i) == m ⊙ max_i(x_i)

本文件守住这两条等价的数学前提，避免将来掩膜改为软边/浮点或累积改为其他
归约时悄悄改变结果。
"""
import sys
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, '.')

from MetLib.collector import MetExporter
from MetLib.metstruct import MDTarget, ModelCfg, RecheckCfg, RuntimeParams
from MetLib.stacker import MaxImgContainer


def make_target(start_frame: int, last_activate_frame: int,
                category: str = "METEOR") -> MDTarget:
    return MDTarget(
        start_frame=start_frame,
        start_time="00:00:01.000",
        end_time="00:00:02.000",
        last_activate_frame=last_activate_frame,
        last_activate_time="00:00:02.000",
        duration=last_activate_frame - start_frame + 1,
        speed=5.0,
        dist=50.0,
        fix_dist=5.0,
        fix_speed=5.0,
        fix_motion_duration=0.5,
        fix_duration=0.5,
        num_pts=3,
        category=category,
        pt1=[10, 10],
        pt2=[40, 40],
        drct_loss=0.1,
        drct_cv=0.05,
        score=0.8,
        real_dist=50.0,
    )


def make_exporter(recheck_mask: Optional[np.ndarray] = None) -> MetExporter:
    recheck_cfg = RecheckCfg(switch=True,
                             model=ModelCfg(name="mock",
                                            weight_path="mock.onnx",
                                            dtype="fp32",
                                            nms=True,
                                            warmup=False,
                                            pos_thre=0.5,
                                            nms_thre=0.45,
                                            multiscale_pred=1,
                                            multiscale_partition=1))
    runtime_param = MagicMock(spec=RuntimeParams)
    runtime_param.positive_category_list = ["METEOR"]
    runtime_param.raw_size = [120, 100]
    runtime_param.runtime_size = [120, 100]
    runtime_param.fps = 30.0
    with patch('MetLib.collector.init_model'):
        return MetExporter(recheck_cfg=recheck_cfg,
                           runtime_param=runtime_param,
                           video_loader=None,
                           logger=MagicMock(),
                           clip_merge_interval=120.0,
                           det_thre=0.7,
                           recheck_mask=recheck_mask)


class TestMaxContainerAccumulation:
    """`np.maximum` 必须与旧的最大值归约逐元素一致。"""

    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
    def test_matches_reference_reduction(self, dtype):
        rng = np.random.default_rng(0)
        frames = [rng.integers(0, 256, (32, 48, 3)).astype(dtype) for _ in range(7)]
        container = MaxImgContainer()
        for frame in frames:
            container.append(frame)
        expected = np.max(np.stack(frames, axis=0), axis=0)
        assert np.array_equal(container.export(), expected)
        assert container.export().dtype == expected.dtype

    def test_single_frame_is_returned_unchanged(self):
        frame = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
        container = MaxImgContainer()
        container.append(frame)
        assert np.array_equal(container.export(), frame)

    def test_does_not_alias_caller_frame(self):
        """原地累积不能改写调用方持有的数组。"""
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        container = MaxImgContainer()
        container.append(frame)
        frame[...] = 255
        assert container.export().max() == 0

    def test_mixed_dtype_falls_back_to_promotion(self):
        """类型不一致时保持旧语义：结果提升为较宽类型，而不是抛错。"""
        container = MaxImgContainer()
        container.append(np.zeros((2, 2, 3), dtype=np.uint8))
        container.append(np.full((2, 2, 3), 1.5, dtype=np.float32))
        assert container.export().dtype == np.float32
        assert container.export().max() == pytest.approx(1.5)

    def test_repeated_appends_match_reference(self):
        rng = np.random.default_rng(7)
        frames = [rng.integers(0, 256, (6, 8, 3)).astype(np.uint8) for _ in range(9)]
        container = MaxImgContainer()
        for frame in frames:
            container.append(frame)
        assert np.array_equal(container.export(), np.max(np.stack(frames), axis=0))


class TestMaskHoistingIdentity:
    """0/1 二值掩膜可以挪到叠帧之后，结果逐像素一致。"""

    @pytest.mark.parametrize("height,width", [(16, 24), (7, 5)])
    def test_max_of_masked_equals_masked_max(self, height, width):
        rng = np.random.default_rng(1)
        frames = [rng.integers(0, 256, (height, width, 3)).astype(np.uint8)
                  for _ in range(6)]
        mask = rng.integers(0, 2, (height, width, 3)).astype(np.uint8)

        per_frame = [frame * mask for frame in frames]
        stacked_then_masked = np.maximum.reduce(frames) * mask
        masked_while_stacking = np.maximum.reduce(per_frame)

        assert np.array_equal(stacked_then_masked, masked_while_stacking)

    def test_non_binary_mask_is_documented_limit(self):
        """软边掩膜仍满足恒等式；该断言用于记录"只要逐像素恒定即可"，而非限制。

        真正的限制是不能用会引入类型回绕的乘法（掩膜须为 0/1 或已归一化）。
        """
        rng = np.random.default_rng(2)
        frames = [rng.integers(0, 256, (8, 8, 3)).astype(np.uint8) for _ in range(4)]
        soft_mask = (rng.integers(0, 11, (8, 8, 3)) / 10.0).astype(np.float32)
        assert np.allclose(np.maximum.reduce(frames) * soft_mask,
                           np.maximum.reduce([f * soft_mask for f in frames]))


class TestExporterAppliesMaskAfterStacking:
    """MetExporter 必须先叠帧、再套一次掩膜，且把掩膜后的图交给模型。"""

    def _run(self, recheck_mask):
        stack = np.full((100, 120, 3), 200, dtype=np.uint8)
        exporter = make_exporter(recheck_mask=recheck_mask)
        exporter.recheck_loader = MagicMock(video_total_frames=10000)
        captured = {}

        def fake_forward(img):
            captured["img"] = img
            # 无匹配框 -> 复检返回 None，避免依赖后续亮度计算
            return np.zeros((0, 4), dtype=np.int64), np.zeros((0, 3), dtype=np.float64)

        exporter.recheck_model.forward.side_effect = fake_forward
        try:
            with patch('MetLib.collector.max_stacker', return_value=stack):
                assert exporter.recheck_single_target(make_target(30, 60)) is None
        finally:
            exporter.export(exporter.END_FLAG, [], 99999, None)
            exporter.export_loop.join(timeout=2.0)
        return stack, captured["img"]

    def test_mask_applied_once_after_stacking(self):
        stack = np.full((100, 120, 3), 200, dtype=np.uint8)
        mask = np.ones((100, 120, 3), dtype=np.uint8)
        mask[:, 60:, :] = 0
        stack_out, model_input = self._run(mask)
        assert np.array_equal(model_input, stack_out * mask)
        # 被掩掉的右半边必须为 0，左半边保持原值
        assert model_input[:, 60:, :].max() == 0
        assert np.all(model_input[:, :60, :] == 200)

    def test_no_mask_passes_stack_through(self):
        stack_out, model_input = self._run(None)
        assert model_input is stack_out
        assert np.all(model_input == 200)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
