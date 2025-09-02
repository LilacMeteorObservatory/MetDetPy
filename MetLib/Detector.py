"""
Detector is the kernel component(?) of the MetDetPy. It detects meteors (and other events) from the given image sequence.

Detector 是MetDetPy的核心组件。其主要在给定的时间窗内检测流星（及其他事件）。

Relation of Detectors in MetDetPy:

                                    |--ClassicDetector
                   |--LineDetector--|
BaseDetector(ABC)--|                |--M3Detector
                   |
                   |--MLDetector
"""

from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Optional, Sequence, Union, cast

import cv2
import numpy as np

from .metlog import BaseMetLog
from .metstruct import BinaryCfg, DLCfg
from .metvisu import (BaseVisuAttrs, DrawRectVisu, ImgVisuAttrs,
                      SquareColorPair, TextColorPair, TextVisu)
from .model import init_model
from .utils import (EMA, PI, SlidingWindow, U8Mat, Uint8EMA, expand_cls_pred,
                    generate_group_interpolate, lineset_nms)

NUM_LINES_TOOMUCH = 500
DEFAULT_INIT_VALUE = 5


class SNR_SW(SlidingWindow):
    """
    ## 检测器使用的滑窗管理器。

    基于滑动窗口维护序列，可以用于产生维护前景图像，参考背景图像，评估信噪比。
    """

    def __init__(self,
                 n: int,
                 mask: U8Mat,
                 est_snr: bool = True,
                 est_area: float = 0,
                 noise_moment: float = 0.99,
                 nz_interval: float = 1) -> None:
        self.est_snr = est_snr
        self.n = n
        self.nz_interval = nz_interval
        # 主滑窗，维护图像
        super().__init__(n,
                         size=mask.shape,
                         dtype=np.uint8,
                         force_int=True,
                         calc_std=False)
        # 需要评估信噪比时，额外的子滑窗
        if self.est_snr:
            self.noise_ema = EMA(momentum=noise_moment, warmup_speed=n)
            self.std_interval = self.nz_interval * n
            self.get_subarea = self.select_subarea(mask, area=est_area)
            sub_h, sub_w = self.std_roi[2] - self.std_roi[0], self.std_roi[
                3] - self.std_roi[1]
            self.sub_sw = SlidingWindow(n,
                                        size=(sub_h, sub_w),
                                        dtype=np.uint8,
                                        force_int=True,
                                        calc_std=False)

    def update(self, new_frame: U8Mat):
        super().update(new_frame)
        self.sub_sw.update(self.get_subarea(new_frame))
        # TODO: 经验公式：当每隔std_interval计算一次时，标准差会存在偏大的情况。结果可除以sqrt(std_interval)以修正数据值。
        # TODO: 利用E(X^2)-E(X)^2的设计会造成10%-20%的性能下降，不符合预期。
        # 在确定合适的经验公式和维护公式之前，不应用该更新。
        # 添加了更快启动的机制。
        # 每std_interval时间更新一次std
        if self.est_snr:
            if self.timer > self.n and self.timer % self.std_interval == 0:
                self.noise_cur_value = np.std(
                    self.sub_sw.sliding_window -
                    np.array(self.sub_sw.mean, dtype=float))
                self.noise_ema.update(self.noise_cur_value)
            elif 1 < self.timer <= self.n:
                self.noise_cur_value = np.std(
                    self.sub_sw.sliding_window[:self.timer] -
                    np.array(self.sub_sw.mean, dtype=float))
                self.noise_ema.update(self.noise_cur_value)

    def select_subarea(self, mask: U8Mat,
                       area: float) -> Callable[[U8Mat], U8Mat]:
        """用于选择一个尽量好的子区域评估STD。

        Args:
            mask (_type_): _description_
            area (float, optional): _description_. Defaults to 0.1.

        Returns:
            Callable: 返回一个可以从numpy中取出局部区域的函数。
        """
        h, w = mask.shape[:2]
        if area == 0:
            self.std_roi = (h // 2, w // 2, 0, 0)
            return lambda img: img

        sub_rate = area**(1 / 2)
        sub_h, sub_w = int(h * sub_rate), int(w * sub_rate)
        x1, y1 = (h - sub_h) // 2, (w - sub_w) // 2
        area = sub_h * sub_w
        light_ratio = np.sum(mask[x1:x1 + sub_h, y1:y1 + sub_w]) / area
        while light_ratio < 1:
            x1 -= 10
            new_ratio = np.sum(mask[x1:x1 + sub_h, y1:y1 + sub_w]) / area
            if new_ratio < light_ratio or x1 < 0:
                x1 += 10
                break
            light_ratio = new_ratio
        self.std_roi = (x1, y1, x1 + sub_h, y1 + sub_w)
        return lambda img: img[x1:x1 + sub_h, y1:y1 + sub_w]

    @property
    def snr(self):
        assert self.est_snr, "cannot get snr with est_snr not applied."
        return self.noise_ema.cur_value


class BaseDetector(metaclass=ABCMeta):
    """An abstract base class for detector.
    
    To implement a detector, you should:xxxxxx 
    TODO: add more explaination.

    VideoLoader class handles the whole video-to-frame process, which should include: 
    1. init - of course, initialize your detector status.
    2. update - update a frame to your detector. your detector may create a stack to store them... or not.
    3. detect - the kernel function that receive no argument and return candidate meteor position series.
    4. visu - return a dict that includes current frame and debug information.

    """

    @abstractmethod
    def __init__(self, *args: Any) -> None:
        pass

    @abstractmethod
    def update(self, new_frame: U8Mat) -> None:
        pass

    @abstractmethod
    def detect(self) -> tuple[Sequence[list[int]], Sequence[list[np.float64]]]:
        pass

    def visu(self) -> list[BaseVisuAttrs]:
        return []


class LineDetector(BaseDetector):
    """基于"二值化-Hough直线检测"的检测器类。
    作为抽象类，并不会产生检测结果。

    LineDetector 输入为需要为grayscale。

    Args:
        window_sec (_type_): 检测器的窗口大小。
        bi_cfg (EasyDict): 二值化参数配置。
        hough_cfg (EasyDict): 直线检测参数配置。

    # 可启用特性：
    自适应阈值
    动态掩模
    动态间隔
    """
    # version I
    sensitivity_func: dict[str, Callable[[float], float]] = {
        #"low": lambda x: 1.5 * x**2 + 4.2, # TODO:not sure, unstable update, needs more sample to validate.
        "low": lambda x: 2.0 * x**2 + 4.4,
        "normal": lambda x: 1.2 * x**2 + 3.6,
        "high": lambda x: 0.9 * x**2 + 3,
    }
    abs_sensitivity = {"high": 3, "normal": 5, "low": 7}
    cv_op = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def __init__(self, window_sec: float, fps: float, mask: U8Mat,
                 num_cls: int, cfg: BinaryCfg, logger: BaseMetLog):
        self.mask = mask
        self.num_cls = num_cls
        self.logger = logger
        self.mask_area = np.sum(self.mask)
        # cfg
        self.bi_cfg = cfg.binary
        self.hough_cfg = cfg.hough_line
        self.dynamic_cfg = cfg.dynamic
        # stack
        self.stack_maxsize = int(window_sec * fps)
        self.stack = SNR_SW(n=self.stack_maxsize,
                            mask=self.mask,
                            est_snr=True,
                            est_area=self.bi_cfg.area,
                            nz_interval=self.bi_cfg.interval)
        # 加载自适应阈值的配置参数
        if self.bi_cfg.adaptive_bi_thre:
            self.std2thre = self.sensitivity_func[self.bi_cfg.sensitivity]
            self.bi_threshold = self.abs_sensitivity[self.bi_cfg.sensitivity]
        else:
            self.bi_threshold = self.bi_cfg.init_value
        self.bi_threshold_float = self.bi_threshold

        # 如果启用动态蒙版（dynamic mask），在此处构建另一个滑窗管理
        if self.dynamic_cfg.dy_mask:
            self.dy_mask_list = SlidingWindow(n=self.stack_maxsize,
                                              size=self.mask.shape,
                                              dtype=np.uint8,
                                              force_int=True)

        # 动态间隔()
        # TODO: 待下线
        self.max_allow_gap = 0.05

        self.visu_param = []

    def detect(self) -> tuple[list[list[int]], list[list[np.float64]]]:
        return [], []

    def update(self, new_frame: U8Mat):
        self.stack.update(new_frame)
        if self.bi_cfg.adaptive_bi_thre and (self.stack.snr != 0):
            self.bi_threshold_float = self.std2thre(self.stack.snr)
            self.bi_threshold = round(self.bi_threshold_float)

    def visu(self):
        return super().visu()

    def calculate_dy_mask(self, act: U8Mat):
        # if "dynamic_mask" is applied, stack and mask dst
        self.dy_mask_list.update(act)
        # TODO: 进一步使Dy_mask稳定作用在持续产生响应的区域，并提供可调整的阈值。
        dy_mask = np.array(self.dy_mask_list.sum
                           <= (self.dy_mask_list.length - 1) * 255,
                           dtype=np.uint8)
        dy_mask = cv2.erode(dy_mask, self.cv_op)
        return np.multiply(act, dy_mask)


class ClassicDetector(LineDetector):
    '''[uzanka-based detector](https://github.com/uzanka/MeteorDetector), in python.
    inspired by uzanka(https://github.com/uzanka)
    '''
    classic_max_size = 4

    def __init__(self, window_sec: float, fps: float, mask: U8Mat,
                 num_cls: int, cfg: BinaryCfg, logger: BaseMetLog):
        # 4帧窗口（硬编码）
        window_sec = self.classic_max_size / fps
        super().__init__(window_sec, fps, mask, num_cls, cfg, logger)

    def detect(self):
        id3, id2, id1, id0 = [
            self.stack.cur_index - i for i in range(self.classic_max_size)
        ]
        sw = self.stack.sliding_window
        # 短于4帧时不进行判定
        if self.stack.timer < self.stack_maxsize:
            return super().detect()

        # 差分2,3帧，二值化，膨胀（高亮为有差异部分）
        diff23 = cast(U8Mat, cv2.absdiff(sw[id2], sw[id3]))
        _, diff23 = cv2.threshold(diff23, self.bi_threshold, 255,
                                  cv2.THRESH_BINARY)
        diff23 = 255 - cv2.dilate(diff23, self.cv_op)  # type: ignore

        ## 用diff23和0,1帧做位与运算（掩模？），屏蔽2,3帧的差一部分
        f1 = cv2.bitwise_and(diff23, sw[id0])
        f2 = cv2.bitwise_and(diff23, sw[id1])

        ## 差分0,1帧，二值化，膨胀（高亮有差异部分）
        dst = cv2.absdiff(f1, f2)
        _, dst = cv2.threshold(dst, self.bi_threshold, 255, cv2.THRESH_BINARY)
        dst = cv2.dilate(dst, self.cv_op)

        # 对0,1帧直线检测
        self.linesp = cv2.HoughLinesP(dst,
                                      rho=1,
                                      theta=PI,
                                      threshold=self.hough_cfg.threshold,
                                      minLineLength=self.hough_cfg.min_len,
                                      maxLineGap=self.hough_cfg.max_gap)

        self.linesp_ext: Sequence[list[int]] = [] if self.linesp is None else self.linesp[:, 0, :]
        # TODO:
        # 1. Classic Detector的可视化接口尚未实现
        # 2. ClassicDetector的输出统一为METEOR判定。可能需要考虑逻辑是否会变更。
        cls_pred = np.zeros((len(self.linesp_ext), self.num_cls))
        cls_pred[:, 0] = 1
        return self.linesp_ext, cls_pred

    def visu(self):
        raise NotImplementedError


class M3Detector(LineDetector):
    """Detector, but version spring.

    主要工作原理： 以X帧为窗口的帧差法 （最大值-中值）。
    
    利用更大的滑窗范围（滑窗中值法）和简化的最大值-阈值方法提升精度与速度
    
    采取了相比原算法更激进的阈值和直线检测限，将假阳性样本通过排除记录流星的算法移除。

    Args:
        stack (_type_): _description_
        threshold (_type_): _description_
        drawing (_type_): _description_

    Returns:
        _type_: _description_
    """

    def __init__(self, window_sec: float, fps: float, mask: U8Mat,
                 num_cls: int, cfg: BinaryCfg, logger: BaseMetLog):
        super().__init__(window_sec, fps, mask, num_cls, cfg, logger)
        self.visu_param: list[BaseVisuAttrs] = [
            DrawRectVisu("results", color="orange"),
            DrawRectVisu("std_roi_area", color="purple"),
            ImgVisuAttrs("mix_bg", weight=0.5, color="yellow"),
            TextVisu("std_value", position="left-top", color="green"),
            TextVisu("bi_value", position="left-top", color="green"),
            TextVisu("lines_num", position="left-top", color="green"),
            TextVisu("area_ratio", position="left-top", color="green"),
            TextVisu("lines_warning", position="left-top", color="red")
        ]

    def detect(self):
        # Preprocessing
        # Mainly calculate diff_img (which basically equals to max-mid)
        light_img = self.stack.max
        diff_img = light_img - self.stack.mean
        diff_img = cv2.medianBlur(diff_img, 3)

        # Post-processing后处理：二值化 + 闭运算
        _, dst = cv2.threshold(diff_img, self.bi_threshold, 255,
                               cv2.THRESH_BINARY)
        #dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, self.cv_op)
        dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, self.cv_op)

        # dynamic_mask 机制
        # if "dynamic_mask" is applied, stack and mask dst
        if self.dynamic_cfg.dy_mask:
            dst = self.calculate_dy_mask(dst)

        self.dst_sum = cast(float,np.sum(dst / 255.) / self.mask_area * 100)
        gap = max(
            0, 1 - self.dst_sum / self.max_allow_gap) * self.hough_cfg.max_gap

        # 核心步骤：直线检测
        linesp = cv2.HoughLinesP(dst,
                                 rho=1,
                                 theta=PI,
                                 threshold=self.hough_cfg.threshold,
                                 minLineLength=self.hough_cfg.min_len,
                                 maxLineGap=gap)
        linesp_ext = np.array([]) if linesp is None else linesp[:, 0, :]

        # 如果产生的响应数目非常多，忽略该帧
        # TODO: 会造成无法响应面积式的现象。需要调整。
        # 下调了阈值以提升面积召回。更合理的版本：
        self.lines_num = len(linesp_ext)
        if self.lines_num > NUM_LINES_TOOMUCH:
            linesp_ext = np.array([])

        # 后处理：对于直线进行质量评定，过滤掉中空比例较大的直线
        # 这一步骤会造成一些暗弱流星的丢失。
        #if len(linesp) > 0:
        #    line_pts = generate_group_interpolate(linesp)
        #    line_score = np.array([
        #        np.sum(dst[line_pt[1], line_pt[0]]) / (len(line_pt[0]) * 255)
        #        for line_pt in line_pts
        #    ])
        #    #for line, line_pt in zip(linesp,line_pts):
        #    #    print(line, line_pt)
        #    #    print(dst[line_pt[1], line_pt[0]])
        #    linesp = linesp[line_score > self.fill_thre]
        #    self.lines_num = len(linesp)

        # 由line预测的结果都划分为不确定（-1），在后处理器中决定类别。
        # TODO: 重整类别格式，前置NMS
        self.linesp_ext = linesp_ext
        self.dst = dst
        # NMS
        # Another TODO: 不确定是否会产生额外的性能开销。
        if len(linesp_ext) > 0:
            linesp_ext, nonline_probs = lineset_nms(linesp_ext)
            self.filtered_line_num = len(linesp_ext)
            cls_pred = np.zeros((self.filtered_line_num, self.num_cls))
            # -1 为OTHERS 所以实际上需要约定该值为OTHERS。
            cls_pred[:, -1] = nonline_probs
            cls_pred[:, 0] = 1 - nonline_probs
        else:
            self.filtered_line_num = 0
            cls_pred = np.zeros((0, self.num_cls))
        return linesp_ext, cls_pred

    def visu(self):
        """ 返回可视化时的所需实际参数，需要与visu_param对应。
        """
        x1, y1, x2, y2 = [-1] * 4
        if getattr(self.stack, "std_roi", None):
            x1, y1, x2, y2 = self.stack.std_roi
        visu_list: list[BaseVisuAttrs] = [
            ImgVisuAttrs("mix_bg", img=self.dst // 255),
            TextVisu(
                "std_value",
                text_list=[TextColorPair(text=f"STD:{self.stack.snr:.4f}")]),
            TextVisu(
                "bi_value",
                text_list=[
                    TextColorPair(
                        text=
                        f"Bi_Threshold: {self.bi_threshold} (rounded from {self.bi_threshold_float:.4f})"
                    )
                ]),
            TextVisu(
                "lines_num",
                text_list=[
                    TextColorPair(
                        text=
                        f"Line num: {self.lines_num} (filtered: {self.filtered_line_num})"
                    )
                ]),
            TextVisu("area_ratio",
                     text_list=[
                         TextColorPair(text=f"Diff Area: {self.dst_sum:.2f}%")
                     ]),
            TextVisu("lines_warning",
                     text_list=[
                         TextColorPair(text="WARNING: TOO MANY LINES!" if self.
                                       lines_num > 10 else "")
                     ]),
            DrawRectVisu(
                "std_roi_area",
                pair_list=[SquareColorPair(dot_pair=([y1, x1], [y2, x2]))])
        ]
        return visu_list


class DiffAreaGuidingDetecor(BaseDetector):
    """差值面积引导的检测器，是从M3Detector在实践中的开发和缺陷综合提出的实验性检测器。
    
    主要工作原理：
    1. 使用 EMA 维护背景图像，使用当前帧作为亮值图像。
    2. 通过直方图统计将当前帧的亮部区分出给定面积比例需要的阈值，记为H_T；通过历史使用的阈值记为E_T。通过卡尔曼滤波计算出当前帧使用的二值化阈值。
    3. 对图像计算二值化，经过简单噪点滤波，动态掩模等机制过滤噪声。
    4. 使用封闭区域查找，将当前帧的前景转换为若干个区域响应。
    5. 重整化响应，输出结果。
    
    主要预期改进点：
    1. 背景图像和亮帧图像的维护更简单。
    2. 通过卡尔曼滤波代替滑窗和噪声估计。（1，2预期能够显著降低计算量）
    3. 通过确保前景的划分比例，相比公式估算，自适应程度有更加显著的提升，对不同噪声的输入灵敏度近似。
    4. 使用区域检测，减少对直线检测的依赖。

    需要解决的问题：
    1. 目前EMA工作方式仍然存在问题，计算差值有逐渐变大的趋势。
    2. 对于多云等复杂天气情况，固定比例分割可能会导致无法正确检测到目标。
    3. 大量可能的噪声的追踪会导致性能下降，需要考虑合适的处理机制。
    
    Args:
        BaseDetector (_type_): _description_
    """

    def __init__(self, window_sec: float, fps: float, mask: U8Mat,
                 num_cls: int, cfg: BinaryCfg, logger: BaseMetLog):
        self.bg_maintainer = Uint8EMA(momentum=(1 - 1 / (window_sec * fps)))
        self.visu_param: list[BaseVisuAttrs] = [
            ImgVisuAttrs("mix_bg", weight=1),
            ImgVisuAttrs("diff_mask", color="yellow", weight=0.5),
            TextVisu("cur_emo_value", position="left-top", color="green")
        ]

    def update(self, new_frame:U8Mat) -> None:
        self.cur_frame = new_frame

    def post_update(self) -> None:
        """DiffAreaGuidingDetecor后置了实际序列的update函数，以便更好计算背景图像与前景的差值。
        """
        self.bg_maintainer.update(self.cur_frame)

    def detect(self):
        if self.bg_maintainer.t == 0:
            # 第一帧仍然填充
            self.bg_maintainer.update(self.cur_frame)
            self.diff_img = np.zeros_like(self.cur_frame)
            return [], []
        neg_value_mask = self.cur_frame < self.bg_maintainer.cur_value + 5
        self.diff_img = self.cur_frame - self.bg_maintainer.cur_value - 5
        self.diff_img[neg_value_mask] = 0
        self.post_update()
        return [], []

    def visu(self):
        ret: list[BaseVisuAttrs] = [
            ImgVisuAttrs("mix_bg", img=self.bg_maintainer.cur_value),
            ImgVisuAttrs("diff_mask", img=self.diff_img),
            TextVisu(
                "cur_emo_value",
                text_list=[
                    TextColorPair(
                        text=f"EMA: {self.bg_maintainer.cur_momentum:.4f}")
                ])
        ]
        return ret


class MLDetector(BaseDetector):
    """Detector with Deep-learning model as backend.
    Support general yolo in onnx format and gives yolo-type output.
    Args:
        BaseDetector (_type_): _description_
    """

    def __init__(self, window_sec: float, fps: float, mask: U8Mat,
                 num_cls: int, cfg: DLCfg, logger: BaseMetLog):
        # stack
        self.mask = mask
        self.num_cls = num_cls
        self.logger = logger
        self.stack_maxsize = int(window_sec * fps)
        self.stack = SlidingWindow(n=self.stack_maxsize,
                                   size=self.mask.shape,
                                   dtype=np.uint8,
                                   force_int=True)
        self.model = init_model(cfg.model, logger=self.logger)
        self.visu_param = [DrawRectVisu("results", color="orange")]

    def update(self, new_frame:U8Mat) -> None:
        self.stack.update(new_frame)

    def detect(self):
        self.result_pos, self.result_cls = self.model.forward(self.stack.max)
        if len(self.result_pos) == 0:
            return super().detect()
        return self.result_pos, expand_cls_pred(self.result_cls)

    def visu(self):
        """ 返回可视化时的所需实际参数，需要与visu_param对应。

        Args:
            bg (np.array): 背景图像
            light (np.array): 响应图像
        """
        visu_list: list[BaseVisuAttrs] = [
            DrawRectVisu("results",
                         pair_list=[
                             SquareColorPair(dot_pair=([x[0], x[1]],
                                                       [x[2], x[3]]))
                             for x in self.result_pos
                         ])
        ]
        return visu_list
