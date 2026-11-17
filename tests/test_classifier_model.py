"""ROI 分类器（MobileNetV3-Large）的初始化契约与可用性测试。

项目内的约定：初始化所需的字段（输入边长、色彩顺序、mean/std、类别表、背景类）
由 `recheck_cfg.gate` 与权重文件自带的 `metadata_props` 共同决定，不再读取外部
导出包的 `metadata.json`/`classes.json`。权重缺失时相关用例跳过。
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, '.')

from MetLib.imgproc import Size, crop_roi, plan_roi_size
from MetLib.metstruct import ClassifierCfg
from MetLib.model import (DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD,
                          ROIClassifierModel, init_classifier,
                          make_classifier_spec)

WEIGHT = Path('weights/mobilenet_v3_large_v2_alpha_fp16.onnx')
HAS_WEIGHT = WEIGHT.exists() and WEIGHT.stat().st_size > 1_000_000
needs_weight = pytest.mark.skipif(not HAS_WEIGHT,
                                  reason='classifier weight missing or LFS pointer')
# 闸门的必填项：switch / weight_path / decision（见 ClassifierCfg）。
WEIGHT_PATH = './weights/mobilenet_v3_large_v2_alpha_fp16.onnx'


def gate_cfg(**overrides) -> ClassifierCfg:
    """默认按 threshold 口径构造；argmax 用例自行覆盖 decision。"""
    params = dict(switch=True, weight_path=WEIGHT_PATH, decision='threshold',
                  p_background_thre=0.5)
    params.update(overrides)
    return ClassifierCfg(**params)


def _fake_backend(input_shape, output_shape=None, class_names=None):
    """伪造 ONNXBackend，用来测"权重形状不合规"这类没有真实权重可用的分支。"""
    backend = MagicMock()
    backend.dtype = np.float16
    backend.input_shape = input_shape
    backend.output_shape = output_shape or [['batch', len(CLASSES)]]
    backend.model_metadata = {'class_names': json.dumps(class_names or CLASSES)}
    return backend


CLASSES = ['METEOR', 'PLANE_SATELLITE', 'RED_SPRITE', 'LIGHTNING', 'JET',
           'RARE_SPRITE', 'SPACECRAFT', 'BUGS', 'BACKGROUND']


class TestClassifierSpec:
    """`make_classifier_spec` 是纯函数，不依赖权重文件。"""

    def test_defaults_follow_imagenet(self):
        spec = make_classifier_spec(input_size=224, class_names=CLASSES)
        assert spec.input_size == Size(224, 224)
        assert spec.color == 'rgb'
        assert spec.mean == DEFAULT_IMAGE_MEAN
        assert spec.std == DEFAULT_IMAGE_STD
        assert spec.class_names == CLASSES

    def test_background_resolved_by_name(self):
        spec = make_classifier_spec(input_size=224, class_names=CLASSES)
        assert spec.background_index == 8
        assert spec.class_names[spec.background_index] == 'BACKGROUND'

    def test_background_id_wins_over_name(self):
        spec = make_classifier_spec(input_size=224, class_names=CLASSES,
                                    background_id=1)
        assert spec.background_index == 1

    def test_no_background_class_is_allowed(self):
        spec = make_classifier_spec(input_size=224, class_names=CLASSES,
                                    background_name=None)
        assert spec.background_index is None

    def test_unknown_background_name_is_rejected(self):
        with pytest.raises(ValueError, match='Background class'):
            make_classifier_spec(input_size=224, class_names=CLASSES,
                                 background_name='SKY')

    def test_background_id_out_of_range_is_rejected(self):
        with pytest.raises(ValueError, match='out of range'):
            make_classifier_spec(input_size=224, class_names=CLASSES,
                                 background_id=len(CLASSES))

    def test_missing_class_names_is_rejected(self):
        with pytest.raises(ValueError, match='class names'):
            make_classifier_spec(input_size=224)

    def test_duplicated_class_names_are_rejected(self):
        with pytest.raises(ValueError, match='not unique'):
            make_classifier_spec(input_size=224,
                                 class_names=['METEOR', 'METEOR'],
                                 background_name=None)

    def test_missing_input_size_is_rejected(self):
        # 尺寸必须由权重给出（标量或 (w, h)）；None / 长度不对都直接报错
        for bad in (None, (320,), (320, 224, 1), (0, 224)):
            with pytest.raises(ValueError, match='ROI size'):
                make_classifier_spec(input_size=bad, class_names=CLASSES)

    def test_unknown_color_order_is_rejected(self):
        with pytest.raises(ValueError, match='color'):
            make_classifier_spec(input_size=224, class_names=CLASSES,
                                 color='yuv')

    def test_mean_std_must_have_three_channels(self):
        with pytest.raises(ValueError, match='mean'):
            make_classifier_spec(input_size=224, class_names=CLASSES,
                                 mean=[0.5, 0.5])
        with pytest.raises(ValueError, match='std'):
            make_classifier_spec(input_size=224, class_names=CLASSES,
                                 std=[0.5])

    def test_non_positive_std_is_rejected(self):
        with pytest.raises(ValueError, match='positive'):
            make_classifier_spec(input_size=224, class_names=CLASSES,
                                 std=[0.2, 0.0, 0.2])


@pytest.fixture(scope='module')
def model():
    if not HAS_WEIGHT:
        pytest.skip('classifier weight missing or LFS pointer')
    return ROIClassifierModel(gate_cfg(providers_key='cpu'))


class TestCropRoi:
    def test_square_size_from_planner(self):
        img = np.zeros((100, 120, 3), dtype=np.uint8)
        # 小框：规划出 min_side（32），裁出来就是 32×32
        size = plan_roi_size(10, 10, 32, min_side=32, scale=1.2)
        assert size == Size(32, 32)
        assert crop_roi(img, [50, 40], [60, 50], size).shape == (32, 32, 3)
        # 大框：边长取 scale × 框宽
        size = plan_roi_size(80, 20, 32, min_side=32, scale=1.2)
        assert size == Size(96, 96)
        assert crop_roi(img, [10, 10], [90, 30], size).shape == (96, 96, 3)

    def test_rect_size(self):
        img = np.zeros((100, 120, 3), dtype=np.uint8)
        roi = crop_roi(img, [50, 40], [60, 50], Size(64, 32))
        assert roi.shape == (32, 64, 3)

    def test_pads_out_of_bounds_with_zero(self):
        img = np.full((40, 40, 3), 200, dtype=np.uint8)
        roi = crop_roi(img, [0, 0], [4, 4], 40)
        assert roi.shape == (40, 40, 3)
        assert roi[0, 0].max() == 0  # 越界部分补黑
        assert roi[20, 20].min() == 200

    def test_inside_region_is_copied_verbatim(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (80, 80, 3), dtype=np.uint8)
        roi = crop_roi(img, [30, 30], [50, 50], 24)
        assert roi.shape == (24, 24, 3)
        # ROI 覆盖源图 [28,52)：中心 20×20 恰好是原框，必须与源图逐像素一致
        assert np.array_equal(roi[2:22, 2:22], img[30:50, 30:50])


@needs_weight
class TestROIClassifierModel:
    @staticmethod
    def random_roi(side=274, seed=0):
        rng = np.random.default_rng(seed)
        return rng.integers(0, 256, (side, side, 3), dtype=np.uint8)

    def test_contract_comes_from_weight(self, model):
        # 类别表与背景索引都来自权重自带的 metadata_props
        assert model.class_names == CLASSES
        assert model.background_index == 8
        assert model.spec.input_size == Size(224, 224)
        assert model.spec.color == 'rgb'

    def test_dtype_auto_follows_fp16_weight(self, model):
        assert np.dtype(model.input_dtype) == np.float16

    def test_preprocess_contract(self, model):
        roi = self.random_roi(274)
        x = model.preprocess(roi)
        assert x.shape == (1, 3, 224, 224)
        assert x.dtype == np.float32
        # (pixel/255 - mean)/std 的量级：灰度 128 附近约 -1 左右
        assert -6 < x.min() and x.max() < 6

    def test_square_and_non_square_roi_both_work(self, model):
        for shape in [(274, 274, 3), (120, 200, 3), (400, 400, 3)]:
            roi = np.random.default_rng(1).integers(0, 256,
                                                    shape,
                                                    dtype=np.uint8)
            assert model.forward(roi).shape == (1, 9)

    def test_probabilities_are_normalized(self, model):
        probs = model.forward(self.random_roi())
        assert probs.shape == (1, 9)
        assert float(probs.sum()) == pytest.approx(1.0, abs=1e-6)
        assert probs.min() >= 0.0

    def test_deterministic(self, model):
        roi = self.random_roi()
        assert np.allclose(model.forward(roi), model.forward(roi), atol=1e-6)

    def test_batch_matches_single(self, model):
        roi = self.random_roi()
        single = model.forward(roi)[0]
        batch = model.forward([roi, roi])
        assert batch.shape == (2, 9)
        assert np.allclose(batch[0], single, atol=1e-5)

    def test_predict_and_background_probability(self, model):
        roi = self.random_roi()
        name, prob = model.predict(roi)
        probs = model.forward(roi)[0]
        assert name == model.class_names[int(np.argmax(probs))]
        assert prob == pytest.approx(float(probs.max()), abs=1e-6)
        assert model.background_probability(roi) == pytest.approx(
            float(probs[model.background_index]), abs=1e-6)

    def test_blank_roi_is_classified_as_background(self, model):
        """全黑 ROI 应被判为背景（训练集外的稳健性下界检查）。"""
        assert model.predict(np.zeros((224, 224, 3), dtype=np.uint8))[0] == \
            'BACKGROUND'

    def test_config_overrides_class_table(self):
        # 配置显式给类别表与背景索引时以配置为准（覆盖权重内置的 class_names）
        names = ['BG'] + [f'C{i}' for i in range(8)]
        model = ROIClassifierModel(
            gate_cfg(providers_key='cpu', classes=names, background_id=0))
        assert model.class_names == names
        assert model.background_index == 0

    def test_non_square_input_is_accepted(self):
        """框架只要求"固定的 (w, h)"；正方形只是自带权重的选择。"""
        backend = _fake_backend(input_shape=[['batch', 3, 224, 320]])
        with patch('MetLib.model.ONNXBackend', return_value=backend):
            model = ROIClassifierModel(gate_cfg(providers_key='cpu'))
        assert model.input_size == Size(320, 224)
        # 预处理按 (w, h) 缩放：输出是 [1, 3, H, W]
        assert model.preprocess(np.zeros((64, 64, 3), np.uint8)).shape == \
            (1, 3, 224, 320)

    def test_dynamic_input_shape_is_rejected(self):
        """输入边长只由权重决定；动态 H/W 直接拒绝，而不是让配置去猜。"""
        backend = _fake_backend(input_shape=[['batch', 3, 'h', 'w']])
        with patch('MetLib.model.ONNXBackend', return_value=backend):
            with pytest.raises(ValueError, match='fixed input'):
                ROIClassifierModel(gate_cfg(providers_key='cpu'))

    def test_unknown_dtype_is_rejected(self):
        with pytest.raises(ValueError, match='dtype'):
            ROIClassifierModel(gate_cfg(providers_key='cpu', dtype='int8'))


class TestGateConfigContract:
    """口径必填、阈值只在 threshold 口径下必填。"""

    @pytest.mark.parametrize('missing', ['switch', 'weight_path', 'decision'])
    def test_behaviour_fields_have_no_default(self, missing):
        params = dict(switch=True, weight_path=WEIGHT_PATH,
                      decision='threshold', p_background_thre=0.5)
        params.pop(missing)
        with pytest.raises(TypeError):
            ClassifierCfg(**params)

    def test_threshold_requires_operating_point(self):
        with pytest.raises(ValueError, match='p_background_thre is required'):
            ClassifierCfg(switch=True, weight_path=WEIGHT_PATH,
                          decision='threshold')

    def test_argmax_does_not_need_operating_point(self):
        cfg = ClassifierCfg(switch=True, weight_path=WEIGHT_PATH,
                            decision='argmax')
        assert cfg.p_background_thre is None

    def test_threshold_must_be_a_probability(self):
        with pytest.raises(ValueError, match=r'\[0, 1\]'):
            gate_cfg(p_background_thre=1.5)

    def test_unknown_decision_is_rejected(self):
        with pytest.raises(ValueError, match='decision'):
            gate_cfg(decision='softmax')

    def test_config_round_trip_through_json(self):
        # dacite/JSON 反序列化同样触发校验（配置写错在启动期就报错）
        from MetLib.metstruct import RecheckCfg
        payload = {
            'switch': True,
            'frame_cache': True,
            'model': {'name': 'mock', 'weight_path': 'm.onnx', 'dtype': 'fp32',
                      'nms': True, 'warmup': False, 'pos_thre': 0.5,
                      'nms_thre': 0.45, 'multiscale_pred': 1,
                      'multiscale_partition': 1},
            'gate': {'switch': True, 'weight_path': WEIGHT_PATH,
                     'decision': 'threshold'},
        }
        with pytest.raises(ValueError, match='p_background_thre is required'):
            RecheckCfg.from_dict(payload)


class TestPlanRoiSize:
    """自适应裁切尺寸：默认与历史一致，开启后把目标占比拉进 ~64 px 区间。"""

    def test_default_matches_legacy_floor(self):
        # 未开启 target_long_side 时，必须与历史规则完全一致
        for long_side in (4, 12, 64, 186, 200, 300):
            expected = max(224, int(np.ceil(1.2 * long_side)))
            assert plan_roi_size(long_side, long_side, 224) == Size(expected,
                                                                   expected)

    def test_small_target_is_magnified(self):
        # 12 px 的目标 → 裁 42 px（放大 5.3×），目标在 224 画布里占 64 px
        size = plan_roi_size(12, 12, 224, target_long_side=64, side_floor=32)
        assert size == Size(42, 42)
        assert 224 * 12 / size.width == pytest.approx(64)

    def test_magnification_is_capped_by_floor(self):
        # 4 px 的目标被下限 32 钳住（最大放大 7×），而不会无限制放大
        size = plan_roi_size(4, 4, 224, target_long_side=64, side_floor=32)
        assert size == Size(32, 32)
        assert 224 / size.width == pytest.approx(7.0)

    def test_large_target_is_not_magnified(self):
        # 目标已经 ≥ 64 px 时不再放大，保持 224 或 1.2× 框
        assert plan_roi_size(64, 64, 224, target_long_side=64) == Size(224, 224)
        assert plan_roi_size(100, 100, 224, target_long_side=64) == Size(224, 224)
        assert plan_roi_size(200, 200, 224, target_long_side=64) == Size(240, 240)

    def test_never_smaller_than_scaled_box(self):
        # 任何情况下都不裁得比 1.2× 框更小
        for long_side in (12, 40, 100, 200, 400):
            size = plan_roi_size(long_side, long_side, 224, target_long_side=64,
                                 side_floor=32)
            assert size.width >= int(np.ceil(1.2 * long_side))
            assert size.height >= int(np.ceil(1.2 * long_side))

    def test_rect_canvas_follows_canvas_aspect(self):
        # 画布 (320, 224)：裁切长宽比跟随画布，长轴仍是目标占比的控制轴
        size = plan_roi_size(12, 12, Size(320, 224), target_long_side=64,
                             side_floor=32)
        assert size == Size(60, 42)
        assert size.width / size.height == pytest.approx(320 / 224, rel=0.02)
        # 方框在矩形画布上也要装得下（短轴不能小于 1.2×框）
        size = plan_roi_size(100, 100, Size(320, 224))
        assert size.height >= 120 and size.width >= 120

    def test_int_and_pair_canvas_agree(self):
        # 标量与二元组等价（OpenCV 风格）
        assert plan_roi_size(12, 12, 224) == plan_roi_size(12, 12, (224, 224))
        assert plan_roi_size(12, 12, Size(224, 224)) == plan_roi_size(12, 12, 224)


class TestAdaptiveRoiConfig:
    def test_default_is_disabled(self):
        assert gate_cfg().target_long_side is None

    def test_invalid_target_long_side_is_rejected(self):
        with pytest.raises(ValueError, match='target_long_side'):
            gate_cfg(target_long_side=0)

    def test_side_floor_must_not_exceed_min_side(self):
        with pytest.raises(ValueError, match='adaptive_side_floor'):
            gate_cfg(target_long_side=64, adaptive_side_floor=300)

    def test_valid_adaptive_config(self):
        cfg = gate_cfg(target_long_side=64, adaptive_side_floor=32)
        assert cfg.target_long_side == 64
        assert cfg.adaptive_side_floor == 32


class TestInitClassifier:
    def test_disabled_returns_none(self):
        assert init_classifier(gate_cfg(switch=False), logger=None) is None
        assert init_classifier(None, logger=None) is None

    @needs_weight
    def test_enabled_builds_model(self):
        model = init_classifier(gate_cfg(providers_key='cpu'), logger=None)
        assert model is not None
        assert model.class_names[0] == 'METEOR'
        assert model.forward(np.zeros((224, 224, 3), np.uint8)).shape == (1, 9)
