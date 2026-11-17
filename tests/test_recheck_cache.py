"""复核帧缓存与增量叠图的单元测试。

对应实现：MetLib/recheck_cache.py（SeriesStackBuilder / RecheckFrameCache）、
MetLib/imgproc.py 的 Transform.tap、MetLib/videoloader.py 的 frame_sink、
MetLib/collector.py 的 offer_stacked_img。
"""
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, '.')

from MetLib.collector import (CACHE_MARGIN_FRAMES, MetExporter,
                               RECHECK_PADDING_SEC)
from MetLib.imgproc import Size, Transform
from MetLib.metstruct import (ClassifierCfg, MDTarget, ModelCfg, RecheckCfg,
                              RuntimeParams)
from MetLib.recheck_cache import RecheckFrameCache, SeriesStackBuilder
from MetLib.videoloader import VanillaVideoLoader
from MetLib.videowrapper import BaseVideoWrapper

SHAPE = (4, 6, 3)


def make_frame(value: int) -> np.ndarray:
    return np.full(SHAPE, value, dtype=np.uint8)


def brute_force(frames: dict[int, np.ndarray],
                start: int, end: int) -> np.ndarray:
    """[start, end) 的最大值叠图。"""
    keys = sorted(k for k in frames if start <= k < end)
    acc = frames[keys[0]].copy()
    for k in keys[1:]:
        np.maximum(acc, frames[k], out=acc)
    return acc


class TestSeriesStackBuilder:
    def test_folds_window_and_ignores_quiet_tail(self):
        builder = SeriesStackBuilder(start_frame=5, last_activate_frame=7,
                                     padding=2)
        # 窗口 = [5-2, 7+2) = [3, 9)
        frames = {i: make_frame(i * 10) for i in range(0, 14)}
        for i, f in frames.items():
            builder.push(i, f)
        stack, refetch = builder.finish()
        assert refetch is None
        assert np.array_equal(stack, brute_force(frames, 3, 9))
        # 静默期（9..13）不影响叠图
        assert stack.max() == 80

    def test_advance_folds_frames_supplied_from_ring(self):
        builder = SeriesStackBuilder(start_frame=5, last_activate_frame=7,
                                     padding=2)
        frames = {i: make_frame(i * 10) for i in range(0, 14)}
        # 先交付 0..11：窗口右端是 9，9..11 落在窗口外，不会被折叠
        for i in range(0, 12):
            builder.push(i, frames[i])
        assert builder.folded_upto == 8
        # 序列续上：窗口右端 9 → 12，9..11 由环形缓冲提供
        builder.advance(10, frames=list(frames.items()))
        stack, refetch = builder.finish()
        assert refetch is None
        assert np.array_equal(stack, brute_force(frames, 3, 12))

    def test_advance_reports_frames_missing_from_ring(self):
        builder = SeriesStackBuilder(start_frame=5, last_activate_frame=7,
                                     padding=2)
        frames = {i: make_frame(i * 10) for i in range(0, 14)}
        for i in range(0, 12):
            builder.push(i, frames[i])
        # 环形缓冲里只剩 9，10/11 已被淘汰 → 需要补读
        builder.advance(10, frames=[(9, frames[9])])
        stack, refetch = builder.finish()
        assert refetch == (10, 11)

    def test_hole_in_delivery_requests_refetch(self):
        builder = SeriesStackBuilder(start_frame=5, last_activate_frame=7,
                                     padding=2)
        frames = {i: make_frame(i * 10) for i in range(0, 14)}
        # 只交付 3,4 与 7..（模拟环形缓冲被冲掉，5、6 丢了）
        for i in (3, 4):
            builder.push(i, frames[i])
        for i in range(7, 11):
            builder.push(i, frames[i])
        stack, refetch = builder.finish(marked=10)
        assert refetch == (5, 6)  # 窗口 [3, 9) 内缺 5、6
        assert np.array_equal(stack, brute_force({3: frames[3], 4: frames[4],
                                                  7: frames[7], 8: frames[8]},
                                                 3, 9))

    def test_advance_within_window_does_not_need_refetch(self):
        builder = SeriesStackBuilder(start_frame=5, last_activate_frame=7,
                                     padding=2)
        frames = {i: make_frame(i * 10) for i in range(0, 14)}
        for i in range(0, 9):  # 只读到 8，窗口右端（9）之后的帧还没到
            builder.push(i, frames[i])
        builder.advance(8, frames=list(frames.items()))  # 窗口右端 9 → 10
        for i in range(9, 14):
            builder.push(i, frames[i])
        stack, refetch = builder.finish()
        assert refetch is None
        assert np.array_equal(stack, brute_force(frames, 3, 10))

    def test_window_end_clamped_by_frame_count(self):
        builder = SeriesStackBuilder(start_frame=5, last_activate_frame=9,
                                     padding=2, frame_count=10)
        # 旧路径的 end = min(L+pad, total-1) = 9（开区间）→ 折叠 [3, 9)
        frames = {i: make_frame(i * 10) for i in range(0, 12)}
        for i, f in frames.items():
            builder.push(i, f)
        stack, _ = builder.finish()
        assert np.array_equal(stack, brute_force(frames, 3, 9))

    def test_ignores_frames_older_than_window(self):
        builder = SeriesStackBuilder(start_frame=5, last_activate_frame=7,
                                     padding=2)
        frames = {i: make_frame(i * 10) for i in range(0, 12)}
        for i, f in frames.items():
            builder.push(i, f)
        stack, _ = builder.finish()
        # 帧 0..2 在窗口起点之外
        assert stack.min() >= 30

    def test_folded_frames_are_copied(self):
        builder = SeriesStackBuilder(start_frame=0, last_activate_frame=0,
                                     padding=1)
        frame = make_frame(50)
        builder.push(0, frame)
        frame[...] = 255  # 调用方改写原帧
        stack, _ = builder.finish()
        assert stack.max() == 50


class TestBucketHistory:
    """分桶历史：补"环形缓冲绕出去"的那一段，且绝不越过窗口右端。"""

    @staticmethod
    def make_cache(capacity_extra=24, span=2, count=64, padding=2):
        cache = RecheckFrameCache()
        cache.configure(padding=padding, extra_capacity=capacity_extra,
                        bucket_span=span, bucket_count=count)
        return cache

    def test_bucket_fills_gap_that_ring_cannot_cover(self):
        # 环只留 6 帧：静默期内 12..31 一定会绕出去，靠桶补
        cache = self.make_cache(capacity_extra=4, span=2, count=32)
        frames = {i: make_frame(i * 5) for i in range(0, 40)}
        builder = cache.attach(1, start_frame=10, last_activate_frame=10)
        for i in range(8, 12):
            cache.push(i, frames[i])
        assert builder.folded_upto == 11
        for i in range(12, 32):
            cache.push(i, frames[i])
        assert cache.payload_frames == cache.capacity == 6
        # 新响应：窗口右端 [8, 20)，12..19 只能来自分桶
        cache.advance(1, 18)
        assert builder.bucketed > 0
        stack, refetch = builder.finish(marked=cache.marked,
                                        frames=cache.snapshot())
        assert refetch is None
        assert np.array_equal(stack, brute_force(frames, 8, 20))

    def test_bucket_never_folds_beyond_window_end(self):
        # 桶 [16,19] 跨过窗口右端 17：只能折叠 [16,18)，不能把 18/19 带进来
        cache = self.make_cache(capacity_extra=24, span=4, count=8)
        frames = {i: make_frame(i * 5) for i in range(0, 40)}
        builder = cache.attach(1, start_frame=10, last_activate_frame=10)
        for i in range(8, 32):
            cache.push(i, frames[i])
        cache.advance(1, 15)  # 窗口 → [8, 17)
        stack, refetch = builder.finish(marked=cache.marked,
                                        frames=cache.snapshot())
        assert refetch is None
        assert np.array_equal(stack, brute_force(frames, 8, 17))
        assert stack.max() == 80  # 帧 16，而不是帧 18/19 的 90/95

    def test_bucket_trim_reports_lost(self):
        # 桶深不足（2 桶 × 2 帧）：被裁掉的帧必须如实报缺口
        cache = self.make_cache(capacity_extra=2, span=2, count=2)
        frames = {i: make_frame(i * 5) for i in range(0, 40)}
        builder = cache.attach(1, start_frame=10, last_activate_frame=10)
        for i in range(8, 12):
            cache.push(i, frames[i])
        for i in range(12, 32):
            cache.push(i, frames[i])
        cache.advance(1, 24)  # 窗口 → [8, 26)，12..25 远超 4 帧桶覆盖
        _, refetch = builder.finish(marked=cache.marked,
                                    frames=cache.snapshot())
        assert refetch is not None
        assert refetch[0] <= 12 and refetch[1] >= 20

    def test_finish_can_use_ring_snapshot(self):
        # 封口时环里还有帧：不该记缺口
        cache = self.make_cache(capacity_extra=24, span=2, count=32)
        frames = {i: make_frame(i * 5) for i in range(0, 40)}
        builder = cache.attach(1, start_frame=10, last_activate_frame=10)
        for i in range(8, 32):
            cache.push(i, frames[i])
        cache.advance(1, 10)          # 窗口不动，12..31 都还在窗口外
        cache.advance(1, 20)          # 窗口 → [8, 22)
        stack, refetch = builder.finish(marked=cache.marked,
                                        frames=cache.snapshot())
        assert refetch is None
        assert np.array_equal(stack, brute_force(frames, 8, 22))

    def test_attach_is_atomic_with_prefill(self):
        """注册与预卷回填必须原子：loader 线程不能插在中间（§〇 的竞态）。"""
        cache = self.make_cache(capacity_extra=24, span=2, count=32)
        cache.configure(padding=2, extra_capacity=24, bucket_span=2,
                        bucket_count=32, frame_count=None)
        for i in range(0, 30):
            cache.push(i, make_frame(i * 5))
        builder = cache.attach(1, start_frame=20, last_activate_frame=22)
        # 预卷 [18, 30] 全部折叠：first=18，limit=24
        assert builder.folded_upto == 23
        assert builder.stack is not None


class TestRecheckFrameCache:
    def test_push_folds_windows_inside_loader_thread(self):
        cache = RecheckFrameCache()
        cache.configure(padding=1, extra_capacity=8)
        builder = cache.attach(1, start_frame=0, last_activate_frame=0)
        # 窗口右端 = 0 + 1 → 帧 0 折叠，帧 2 落在窗口外
        cache.push(0, make_frame(10))
        assert builder.folded_upto == 0
        cache.push(2, make_frame(30))
        assert builder.folded_upto == 0      # 未被折叠
        assert cache.payload_frames == 2

    def test_attach_prefills_from_ring(self):
        cache = RecheckFrameCache()
        cache.configure(padding=2, frame_count=100, extra_capacity=6)
        frames = {i: make_frame(i * 10) for i in range(0, 12)}
        for i, f in frames.items():
            cache.push(i, f)
        builder = cache.attach(1, start_frame=8, last_activate_frame=9)
        for i in range(10, 14):
            cache.push(i, frames.setdefault(i, make_frame(i * 10)))
        stack, refetch = builder.finish(marked=cache.marked)
        assert refetch is None
        # 窗口 = [6, 11)：预卷 6,7 由环形缓冲回填
        assert np.array_equal(stack, brute_force(frames, 6, 11))

    def test_monotonic_guard_drops_rewound_reads(self):
        cache = RecheckFrameCache()
        cache.configure(padding=1, extra_capacity=4)
        cache.push(0, make_frame(10))
        cache.push(5, make_frame(50))
        cache.push(3, make_frame(30))  # 反向（例如曝光估计重读）
        assert [i for i, _ in cache._ring] == [0, 5]

    def test_detach_removes_builder(self):
        cache = RecheckFrameCache()
        cache.configure(padding=1, extra_capacity=4)
        cache.attach(7, start_frame=0, last_activate_frame=0)
        assert cache.active_count == 1
        assert cache.detach(7) is not None
        assert cache.active_count == 0
        assert cache.detach(7) is None

    def test_reset_clears_state(self):
        cache = RecheckFrameCache()
        cache.configure(padding=1, extra_capacity=4)
        cache.push(0, make_frame(1))
        builder = cache.attach(1, start_frame=0, last_activate_frame=0)
        cache.reset()
        assert cache.active_count == 0
        assert cache.payload_frames == 0
        stack, refetch = builder.finish()
        assert refetch is None
        # 已注销的构建器被封口：不再接收新帧
        cache.push(1, make_frame(200))
        assert np.array_equal(builder._stack, stack)


class TestTransformTap:
    def test_tap_passes_through_and_observes(self):
        seen = []
        transform = Transform()
        transform.tap(lambda img: (seen.append(int(img.max())), img)[1])
        img = make_frame(7)
        out = transform.exec_transform(img)
        assert out is img
        assert seen == [7]


class FakeWrapper(BaseVideoWrapper):
    """最小 wrapper：不真的解码，只用于验证 loader 的旁路接线。"""

    def __init__(self, video_name, hwaccel=None):
        pass

    @property
    def fps(self):
        return 25.0

    @property
    def num_frames(self):
        return 100

    @property
    def size(self):
        return [1920, 1080]

    @property
    def backend_name(self):
        return 'FakeWrapper'

    def set_to(self, frame_num):
        return True

    def get_video_pos(self):
        return 0

    def read(self):
        return False, None


class TestLoaderTapWiring:
    def make_loader(self, **kwargs):
        return VanillaVideoLoader(FakeWrapper, 'fake.mp4', None, 960,
                                  grayscale=True, exp_option='real-time',
                                  **kwargs)

    def test_tap_advances_index_from_start_frame(self):
        got = []
        loader = self.make_loader(frame_sink=lambda i, f: got.append(i))
        loader.start()
        for _ in range(3):
            loader.preprocess.exec_transform(make_frame(1))
        assert got == [loader.start_frame,
                       loader.start_frame + 1,
                       loader.start_frame + 2]

    def test_no_sink_means_no_tap_step(self):
        loader = self.make_loader()
        img = make_frame(3)
        out = loader.preprocess.exec_transform(img)
        assert out.shape == (540, 960)  # 仍然完成 resize + 灰度
        assert all(getattr(fn, '__name__', '') != 'tap'
                   for fn, _ in loader.preprocess.transform)

    def test_debayer_disables_tap(self):
        got = []
        loader = VanillaVideoLoader(FakeWrapper, 'fake.mp4', None, 960,
                                    grayscale=True, debayer=True,
                                    exp_option='real-time',
                                    frame_sink=lambda i, f: got.append(i))
        loader.start()
        loader.preprocess.exec_transform(make_frame(1))
        assert got == []


def make_exporter(recheck_mask=None, gate=None) -> MetExporter:
    recheck_cfg = RecheckCfg(switch=True,
                             gate=gate,
                             yolo_stack_source='cache',
                             model=ModelCfg(name='mock',
                                            weight_path='mock.onnx',
                                            dtype='fp32',
                                            nms=True,
                                            warmup=False,
                                            pos_thre=0.5,
                                            nms_thre=0.45,
                                            multiscale_pred=1,
                                            multiscale_partition=1))
    runtime_param = MagicMock(spec=RuntimeParams)
    runtime_param.positive_category_list = ["METEOR"]
    runtime_param.raw_size = [120, 100]
    runtime_param.runtime_size = [60, 50]
    runtime_param.fps = 25.0
    with patch('MetLib.collector.init_model'), \
            patch('MetLib.collector.init_classifier', return_value=None):
        return MetExporter(recheck_cfg=recheck_cfg,
                           runtime_param=runtime_param,
                           video_loader=MagicMock(video_total_frames=10000),
                           logger=MagicMock(),
                           clip_merge_interval=120.0,
                           det_thre=0.7,
                           recheck_mask=recheck_mask)


def make_target() -> MDTarget:
    return MDTarget(start_frame=10,
                    start_time="00:00:00.400",
                    end_time="00:00:01.000",
                    last_activate_frame=20,
                    last_activate_time="00:00:00.800",
                    duration=11,
                    speed=5.0,
                    dist=50.0,
                    fix_dist=5.0,
                    fix_speed=5.0,
                    fix_motion_duration=0.5,
                    fix_duration=0.5,
                    num_pts=3,
                    category="METEOR",
                    pt1=[10, 10],
                    pt2=[40, 40],
                    drct_loss=0.1,
                    drct_cv=0.05,
                    score=0.8,
                    real_dist=50.0)


class TestExporterUsesOfferedStack:
    def test_offered_stack_skips_refetch_and_gets_masked(self):
        mask = np.zeros((50, 60, 3), dtype=np.uint8)
        mask[:, :30] = 1
        exporter = make_exporter(recheck_mask=mask)
        target = make_target()
        stack = np.full((50, 60, 3), 200, dtype=np.uint8)
        exporter.offer_stacked_img(id(target), stack)
        captured = {}

        def forward(img):
            captured['img'] = img
            return np.zeros((0, 4), dtype=np.int64), np.zeros((0, 3))

        exporter.recheck_model.forward.side_effect = forward
        try:
            with patch('MetLib.collector.max_stacker') as stacker:
                assert exporter.recheck_single_target(target) is None
                stacker.assert_not_called()
        finally:
            exporter.export(exporter.END_FLAG, [], 99999, None)
            exporter.export_loop.join(timeout=2.0)
        img = captured['img']
        assert img[:, :30].max() == 200
        assert img[:, 30:].max() == 0
        # 取用后即释放
        assert id(target) not in exporter._pending_stacks

    def test_missing_stack_falls_back_to_refetch(self):
        exporter = make_exporter()
        target = make_target()
        image = np.zeros((50, 60, 3), dtype=np.uint8)
        exporter.recheck_model.forward.return_value = (
            np.zeros((0, 4), dtype=np.int64), np.zeros((0, 3)))
        try:
            with patch('MetLib.collector.max_stacker',
                       return_value=image) as stacker:
                exporter.recheck_single_target(target)
                stacker.assert_called_once()
        finally:
            exporter.export(exporter.END_FLAG, [], 99999, None)
            exporter.export_loop.join(timeout=2.0)

    def test_gap_is_patched_before_use_not_reread_whole_window(self):
        # 有缺口时只补 [lost_from, lost_to+1]，不整窗重读
        exporter = make_exporter()
        target = make_target()
        stack = np.full((50, 60, 3), 100, dtype=np.uint8)
        patch_img = np.zeros((50, 60, 3), dtype=np.uint8)
        patch_img[:, :10] = 200          # 缺口区里有更亮的一帧
        captured = {}
        exporter.recheck_model.forward.side_effect = lambda img: (
            captured.setdefault('img', img),
            (np.zeros((0, 4), np.int64), np.zeros((0, 3))))[1]
        exporter.offer_stacked_img(id(target), stack, (100, 110))
        try:
            with patch('MetLib.collector.max_stacker',
                       return_value=patch_img) as stacker:
                exporter.recheck_single_target(target)
                stacker.assert_called_once()
                assert stacker.call_args.kwargs['start_frame'] == 100
                assert stacker.call_args.kwargs['end_frame'] == 111
        finally:
            exporter.export(exporter.END_FLAG, [], 99999, None)
            exporter.export_loop.join(timeout=2.0)
        # 补丁已并入闸门/YOLO 共用的那张图
        img = captured['img']
        assert img[:, :10].max() == 200
        assert img[:, 20:].max() == 100

    def test_bad_patch_falls_back_to_full_reread(self):
        exporter = make_exporter()
        target = make_target()
        stack = np.full((50, 60, 3), 100, dtype=np.uint8)
        wrong = np.full((40, 40, 3), 200, dtype=np.uint8)   # 形状不符
        exporter.recheck_model.forward.return_value = (
            np.zeros((0, 4), dtype=np.int64), np.zeros((0, 3)))
        exporter.offer_stacked_img(id(target), stack, (100, 110))
        try:
            with patch('MetLib.collector.max_stacker',
                       side_effect=[wrong, wrong]) as stacker:
                exporter.recheck_single_target(target)
                assert stacker.call_count == 2      # 补丁 → 退化为整窗重读
        finally:
            exporter.export(exporter.END_FLAG, [], 99999, None)
            exporter.export_loop.join(timeout=2.0)

    def test_incomplete_stack_skips_gate_and_counts(self):
        exporter = make_gate_exporter(CONFIDENT_BG_PROBS, decision='threshold')
        target = make_target()
        stack = np.full((50, 60, 3), 100, dtype=np.uint8)
        exporter.recheck_model.forward.return_value = (
            np.zeros((0, 4), dtype=np.int64), np.zeros((0, 3)))
        exporter.offer_stacked_img(id(target), stack, (100, 110))
        try:
            with patch('MetLib.collector.max_stacker', return_value=None):
                exporter.recheck_single_target(target)
        finally:
            exporter.export(exporter.END_FLAG, [], 99999, None)
            exporter.export_loop.join(timeout=2.0)
        # 补丁失败 + 整窗重读失败 → 不完整：闸门不判定，只记账
        assert exporter.gate_skipped_incomplete == 1
        assert exporter.recheck_model.forward.call_count == 1


class FakeGateModel:
    """只实现闸门用到的接口（forward / background_probability / 类别表）。"""

    class_names = ['METEOR', 'RED_SPRITE', 'SPACEWASTE', 'PLANE_SATELLITE',
                   'SPACECRAFT', 'HUMAN', 'ANIMAL', 'BRIGHT', 'BACKGROUND']
    background_index = 8

    def __init__(self, probs):
        self.probs = np.asarray(probs, dtype=np.float64)

    # 与 `ROIClassifierModel.input_size` 对齐：闸门用它换算裁切边长
    input_size = 224

    def forward(self, roi):
        return self.probs[None, :]

    def background_probability(self, roi):
        return float(self.probs[self.background_index])


# 背景概率 0.42、流星 0.30、其余分散：阈值口径放行，argmax 口径拒绝。
SPREAD_PROBS = [0.30, 0.02, 0.02, 0.10, 0.02, 0.02, 0.02, 0.08, 0.42]
CONFIDENT_BG_PROBS = [0.03, 0.01, 0.01, 0.02, 0.01, 0.005, 0.005, 0.01, 0.90]
CONFIDENT_METEOR_PROBS = [0.85, 0.02, 0.01, 0.06, 0.01, 0.01, 0.01, 0.02, 0.01]
# 闸门必填项之一（这里不真正加载权重，路径只占位）。
GATE_WEIGHT = './weights/mobilenet_v3_large_v2_alpha_fp16.onnx'


def make_gate_exporter(probs, decision='threshold', thre=0.5):
    params = dict(switch=True, weight_path=GATE_WEIGHT, decision=decision)
    if decision == 'threshold':
        params['p_background_thre'] = thre       # argmax 口径不消费阈值
    exporter = make_exporter()
    exporter.gate_cfg = ClassifierCfg(**params)
    exporter.gate_model = FakeGateModel(probs)
    return exporter


def shutdown_exporter(exporter):
    exporter.export(exporter.END_FLAG, [], 99999, None)
    exporter.export_loop.join(timeout=2.0)


class TestROIGateCrop:
    """闸门裁切：默认沿用 224 地板；开启自适应后按目标长边放大。"""

    def crop_side(self, cfg_overrides, box=((10, 10), (22, 22))):
        exporter = make_gate_exporter(SPREAD_PROBS, decision='threshold')
        for key, value in cfg_overrides.items():
            setattr(exporter.gate_cfg, key, value)
        target = make_target()
        target.pt1 = list(box[0])
        target.pt2 = list(box[1])
        captured = {}
        import MetLib.collector as collector

        original = collector.crop_roi

        def spy(img, pt1, pt2, size, **kwargs):
            captured['size'] = size
            return original(img, pt1, pt2, size, **kwargs)

        try:
            with patch('MetLib.collector.crop_roi', side_effect=spy):
                exporter._apply_roi_gate(target,
                                         np.full((200, 200, 3), 200, np.uint8))
        finally:
            shutdown_exporter(exporter)
        return captured

    def test_default_keeps_legacy_floor(self):
        captured = self.crop_side({'target_long_side': None})
        assert captured['size'] == Size(224, 224)

    def test_adaptive_magnifies_small_target(self):
        # 12 px 框 → 裁 42 px（目标在 224 画布里占 64 px）
        captured = self.crop_side({'target_long_side': 64,
                                   'adaptive_side_floor': 32})
        assert captured['size'] == Size(42, 42)

    def test_adaptive_leaves_large_target_alone(self):
        captured = self.crop_side({'target_long_side': 64,
                                   'adaptive_side_floor': 32},
                                  box=((10, 10), (150, 150)))
        assert captured['size'] == Size(224, 224)


class TestROIGateDecision:
    """闸门两种判定口径：threshold（默认）与 argmax（更严格）。"""

    def gate(self, probs, decision, target=None, thre=0.5):
        exporter = make_gate_exporter(probs, decision=decision, thre=thre)
        try:
            return exporter._apply_roi_gate(
                target or make_target(),
                np.full((50, 60, 3), 200, dtype=np.uint8))
        finally:
            shutdown_exporter(exporter)

    def test_threshold_mode_keeps_spread_candidate(self):
        # p_background = 0.42 < 0.5：阈值口径放行
        assert self.gate(SPREAD_PROBS, 'threshold') is False

    def test_argmax_mode_rejects_spread_candidate(self):
        # 同一个候选：背景仍是 top-1，argmax 口径拒（比值更严格、只多拒）
        assert self.gate(SPREAD_PROBS, 'argmax') is True

    def test_both_modes_agree_on_confident_background(self):
        assert self.gate(CONFIDENT_BG_PROBS, 'threshold') is True
        assert self.gate(CONFIDENT_BG_PROBS, 'argmax') is True

    def test_both_modes_agree_on_confident_meteor(self):
        assert self.gate(CONFIDENT_METEOR_PROBS, 'threshold') is False
        assert self.gate(CONFIDENT_METEOR_PROBS, 'argmax') is False

    def test_threshold_lower_than_background_prob_still_rejects(self):
        assert self.gate(CONFIDENT_BG_PROBS, 'threshold', thre=0.95) is False

    def test_argmax_rejection_skips_yolo(self):
        exporter = make_gate_exporter(SPREAD_PROBS, decision='argmax')
        target = make_target()
        exporter.offer_stacked_img(id(target),
                                   np.full((50, 60, 3), 200, dtype=np.uint8))
        try:
            assert exporter.recheck_single_target(target) is None
            exporter.recheck_model.forward.assert_not_called()
            assert exporter.recheck_model_call_count == 0
        finally:
            shutdown_exporter(exporter)

    def test_threshold_mode_still_calls_yolo_for_spread_candidate(self):
        exporter = make_gate_exporter(SPREAD_PROBS, decision='threshold')
        target = make_target()
        exporter.offer_stacked_img(id(target),
                                   np.full((50, 60, 3), 200, dtype=np.uint8))
        exporter.recheck_model.forward.return_value = (
            np.zeros((0, 4), dtype=np.int64), np.zeros((0, 3)))
        try:
            exporter.recheck_single_target(target)
            assert exporter.recheck_model.forward.call_count == 1
        finally:
            shutdown_exporter(exporter)

    def test_invalid_decision_rejected(self):
        # 口径校验在 ClassifierCfg 构造期完成，collector 不再重复判断
        with pytest.raises(ValueError, match='decision'):
            ClassifierCfg(switch=True, weight_path=GATE_WEIGHT,
                          decision='softmax')


class TestCollectorOwnsRecheckResources:
    """掩膜加载与帧缓存配置都由 collector 负责，主进程只做接线。"""

    def make_cfg_and_param(self):
        from MetLib.metstruct import MainDetectCfg, RuntimeParams
        collector_cfg = MainDetectCfg.from_json_file(
            'config/m3det_normal.json').collector
        runtime_param = RuntimeParams(fps=25.0,
                                      exp_frame=4,
                                      eq_fps=6.25,
                                      eq_int_fps=6,
                                      exp_time=0.16,
                                      runtime_size=[960, 540],
                                      raw_size=[3840, 2160],
                                      positive_category_list=['METEOR'])
        return collector_cfg, runtime_param

    def test_loads_mask_and_configures_cache(self):
        from MetLib.collector import MeteorCollector
        collector_cfg, runtime_param = self.make_cfg_and_param()
        cache = RecheckFrameCache()
        loader = MagicMock(video_total_frames=317, maxsize=32)
        with patch('MetLib.collector.MetExporter'), \
                patch('MetLib.collector.load_mask') as load_mask:
            collector_obj = MeteorCollector(collector_cfg, runtime_param, loader,
                                            MagicMock(), mask_name='mask.png',
                                            frame_cache=cache)
        load_mask.assert_called_once_with('mask.png',
                                          runtime_param.runtime_size,
                                          grayscale=False)
        # 环深 = 领先(队列原始帧数+1) + max(后视需求, 桶边界+余量)
        # 注意 lead 用 maxsize 而不是 maxsize×exp_frame：队列装的是原始帧，
        # 合并发生在消费端 pop()（见 MetLib/videoloader.py:597-621）。
        assert cache.padding == int(RECHECK_PADDING_SEC * runtime_param.fps)
        span = max(1, cache.padding)
        lead = 32 + 1
        rear_need = runtime_param.exp_frame + 1
        assert cache.extra_capacity == lead + max(
            rear_need, span + CACHE_MARGIN_FRAMES)
        # 桶要覆盖 "静默期 + 领先量 + 边界桶"
        depth = collector_obj.max_acti_frame + lead + cache.padding
        assert cache.bucket_span == span
        assert cache.bucket_count == -(-depth // span) + 1
        assert cache.frame_count == 317
        assert cache.payload_frames == 0  # 构造时已 reset

    def test_producer_maxsize_override_wins_over_loader(self):
        from MetLib.collector import MeteorCollector
        collector_cfg, runtime_param = self.make_cfg_and_param()
        cache = RecheckFrameCache()
        # 生产者（主 loader）队列只有 8 深：环应当明显变小
        with patch('MetLib.collector.MetExporter'), \
                patch('MetLib.collector.load_mask'):
            MeteorCollector(collector_cfg, runtime_param,
                            MagicMock(video_total_frames=317, maxsize=32),
                            MagicMock(), frame_cache=cache,
                            producer_maxsize=8)
        span = max(1, cache.padding)
        assert cache.extra_capacity == 9 + max(
            runtime_param.exp_frame + 1, span + CACHE_MARGIN_FRAMES)

    def test_explicit_mask_wins_over_mask_name(self):
        from MetLib.collector import MeteorCollector
        collector_cfg, runtime_param = self.make_cfg_and_param()
        explicit = np.ones((540, 960, 3), dtype=np.uint8)
        with patch('MetLib.collector.MetExporter'), \
                patch('MetLib.collector.load_mask') as load_mask:
            MeteorCollector(collector_cfg, runtime_param, MagicMock(),
                            MagicMock(), recheck_mask=explicit,
                            mask_name='mask.png')
        load_mask.assert_not_called()

    def test_no_frame_cache_means_legacy_path(self):
        from MetLib.collector import MeteorCollector
        collector_cfg, runtime_param = self.make_cfg_and_param()
        with patch('MetLib.collector.MetExporter'):
            collector = MeteorCollector(collector_cfg, runtime_param,
                                        MagicMock(), MagicMock())
        assert collector.frame_cache is None


class TestConfigDefaults:
    def test_yolo_stack_source_defaults_to_reread(self):
        # 两条路径实测不等价，缺省不能走未验证的 cache
        default = RecheckCfg.__dataclass_fields__['yolo_stack_source'].default
        assert default == 'reread'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
