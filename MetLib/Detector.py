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

from .feature import calc_roi_gradient, crop_with_box
from .metlog import BaseMetLog
from .metstruct import BinaryCfg, Box, BrightnessCfg, DLCfg
from .metvisu import (BaseVisuAttrs, DrawRectVisu, ImgVisuAttrs,
                      SquareColorPair, TextColorPair, TextVisu)
from .model import init_model
from .profiling import profiled
from .utils import (EMA, PI, SlidingWindow, U8Mat, Uint8EMA, expand_cls_pred,
                    generate_group_interpolate, get_name2id, get_num_class,
                    lineset_nms)

NUM_LINES_TOOMUCH = 500
DEFAULT_INIT_VALUE = 5


class NoiseSW(SlidingWindow):
    """## 检测器使用的滑窗管理器。

    基于滑动窗口维护序列，可以用于产生维护前景图像，参考背景图像，评估噪声水平。

    ## 噪声评估（est_noise=True）

    在选定的子区域（且可抽取）上维护逐像素的时间二阶矩（整数累加器），结合主
    滑窗已维护的一阶矩，得到池化的噪声标准差：

        sigma = sqrt( mean_p( Σx²/L - (Σx/L)² ) )

    即「逐像素时间方差在子区域内池化」。
    
    实现方式：

    * 累加器只铺在子区域上（默认再 2×2 抽取），成本约为全帧的 1/40；
    * 一阶矩复用主滑窗的 `sum`，每帧只需维护 Σx²（平方 + 入环 + 加减）；
    * 开方只按 `noise_interval` 采样执行；整数累加、浮点评估
    * 轻量计算，密集采样避免「固定采样步长与编码 GOP 周期拍频」造成的相位锁定。

    噪声平滑使用 EMA，时间常数由 `noise_window_sec` 指定。

    冷启动：窗口未填满时**不产生估计**（`noise_sigma` 保持 0，调用方继续使用固定
    阈值）；窗口填满后在 `noise_warmup` 个样本内把动量抬升到稳态值。

    Args:
        n (int): 滑窗帧数。
        mask (U8Mat): 运行时掩模，用于选择噪声评估子区域。
        fps (float): 送入滑窗的帧率（VideoLoader 的 eq_fps），用于把采样步长换算
            为秒，仅影响时间常数标定。
        est_noise (bool): 是否评估噪声水平。
        est_area (float): 噪声评估子区域占整帧的面积比例；0 表示整帧。
        noise_interval (float): 噪声评估的采样步长（帧）。1 表示每帧评估。
        noise_window_sec (float): 噪声 EMA 的时间常数（秒）。
        noise_warmup (int): 动量从 0 抬升到稳态值所跨的样本数。
        noise_decimate (int): 子区域抽取步长，1 表示不抽取。
    """

    def __init__(self,
                 n: int,
                 mask: U8Mat,
                 fps: float = 1.0,
                 est_noise: bool = True,
                 est_area: float = 0,
                 noise_interval: float = 1,
                 noise_window_sec: float = 60.0,
                 noise_warmup: int = 10,
                 noise_decimate: int = 2) -> None:
        self.est_noise = est_noise
        self.n = n
        self.noise_interval = max(1, int(noise_interval))
        self.noise_window_sec = float(noise_window_sec)
        self.noise_decimate = max(1, int(noise_decimate))
        # 主滑窗，维护图像
        super().__init__(n,
                         size=mask.shape,
                         dtype=np.uint8,
                         force_int=True,
                         calc_std=False)
        # 需要评估噪声水平时，在子区域上维护累加器
        if self.est_noise:
            self.get_subarea = self.select_subarea(mask, area=est_area)
            roi_shape = self.get_subarea(
                np.zeros(mask.shape, dtype=np.uint8)).shape
            self.roi_shape = roi_shape
            # 一阶矩不必自己维护：主滑窗的 `sum` 已经是逐像素的 Σx（uint32），
            # 取它的（抽取）子区域即可，见 `_measure`。这里只维护 Σx²。
            self._noise_sumsq = np.zeros(roi_shape, dtype=np.int64)
            # 环形缓存：已平方的（抽取后）子区域，用于环形移除。
            self._sq_ring = np.zeros((n, ) + roi_shape, dtype=np.int32)
            self._sq_scratch = np.empty(roi_shape, dtype=np.int64)
            self._buf_mu = np.empty(roi_shape, dtype=np.float64)
            self._buf_var = np.empty(roi_shape, dtype=np.float64)
            self._buf_sq = np.empty(roi_shape, dtype=np.float64)
            # 采样周期（秒）与 EMA 动量：保证时间常数为 noise_window_sec 秒。
            period = self.noise_interval / max(float(fps), 1e-9)
            momentum = min(
                max(1.0 - period / max(self.noise_window_sec, 1e-9), 0.0), 1.0)
            warmup_speed = (1.0 / (noise_warmup * (1.0 - momentum))
                            if noise_warmup > 0 and momentum < 1.0 else 0.0)
            self.noise_ema = EMA(momentum=momentum,
                                 warmup_speed=warmup_speed)
            self.noise_sample = 0.0
            self.noise_sample_count = 0

    @profiled("m3.update.stack")
    def update(self, new_frame: U8Mat):
        if self.est_noise:
            self._accumulate(new_frame)
        super().update(new_frame)
        # 窗口未填满时不产生估计：此时窗口均值尚不可信，短窗估计会系统性偏低。
        if self.est_noise and self.timer >= self.n and (
            (self.timer - self.n) % self.noise_interval == 0):
            self.noise_sample = self._measure()
            self.noise_sample_count += 1
            self.noise_ema.update(self.noise_sample)

    def _accumulate(self, new_frame: U8Mat):
        """把新帧的（抽取后）子区域计入二阶矩，并移出被覆盖的旧帧。
        """
        roi = self.get_subarea(new_frame)
        slot = self.timer % self.n
        np.square(roi, out=self._sq_scratch, dtype=np.int64)
        if self.timer >= self.n:
            self._noise_sumsq -= self._sq_ring[slot]
        self._sq_ring[slot] = self._sq_scratch
        self._noise_sumsq += self._sq_scratch

    def _measure(self) -> float:
        """由累加器得到当前窗口的噪声标准差（逐像素时间方差池化）。"""
        inv = 1.0 / self.length
        # Σx 直接取主滑窗的 sum 子区域
        np.multiply(self.get_subarea(self.sum),
                    inv,
                    out=self._buf_mu,
                    dtype=np.float64)
        np.multiply(self._noise_sumsq, inv, out=self._buf_var,
                    dtype=np.float64)
        np.square(self._buf_mu, out=self._buf_sq)
        np.subtract(self._buf_var, self._buf_sq, out=self._buf_var)
        np.maximum(self._buf_var, 0.0, out=self._buf_var)
        return float(np.sqrt(self._buf_var.mean()))

    def select_subarea(self, mask: U8Mat,
                       area: float) -> Callable[[U8Mat], U8Mat]:
        """用于选择一个尽量好的子区域评估噪声水平。

        返回一个从图像中取出（抽取后的）噪声评估子区域的函数；完整分辨率的区域
        写入 `self.std_roi`，供可视化使用。

        Args:
            mask (U8Mat): 运行时掩模，1 表示有效天空。
            area (float): 子区域占整帧的面积比例；0 或负数表示整帧。

        Returns:
            Callable[[U8Mat], U8Mat]: 子区域切片函数。
        """
        h, w = mask.shape[:2]
        dec = self.noise_decimate
        if area <= 0:
            self.std_roi = (0, 0, h, w)
            return lambda img: img[::dec, ::dec]

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
        return lambda img: img[x1:x1 + sub_h:dec, y1:y1 + sub_w:dec]

    @property
    def noise_sigma(self):
        """当前噪声水平（DN）。窗口未填满或未启用评估时为 0。"""
        assert self.est_noise, "noise_sigma requires est_noise."
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


class MockDetector(BaseDetector):
    """No-op detector for feed-pipeline benchmarks (gray or color input).

    Keeps no image references and performs no image processing. Subclass this
    detector to introduce controlled work in update/detect for experiments.
    The standard detector config is accepted but unused.
    """

    def __init__(self, window_sec: float, fps: float, mask: U8Mat, cfg: Any,
                 logger: BaseMetLog):
        self.frames_seen = 0

    def update(self, new_frame: U8Mat) -> None:
        self.frames_seen += 1

    def detect(self) -> tuple[list[list[int]], list[list[np.float64]]]:
        return [], []


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
                 cfg: BinaryCfg, logger: BaseMetLog):
        self.num_cls = get_num_class()
        self.mask = mask
        self.logger = logger
        self.fps = fps
        self.mask_area = np.sum(self.mask)
        # cfg
        self.bi_cfg = cfg.binary
        self.hough_cfg = cfg.hough_line
        self.dynamic_cfg = cfg.dynamic
        # stack
        self.stack_maxsize = int(window_sec * fps)
        self.stack = NoiseSW(n=self.stack_maxsize,
                             mask=self.mask,
                             fps=fps,
                             est_noise=self.bi_cfg.adaptive_bi_thre,
                             est_area=self.bi_cfg.area,
                             noise_interval=self.bi_cfg.interval)
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
                                              force_int=True,
                                              calc_max=False)
            self._dy_mask_buffer = np.empty(self.mask.shape, dtype=np.uint8)

        # 动态间隔()
        # TODO: 待下线
        self.max_allow_gap = 0.05

    def detect(self) -> tuple[list[list[int]], list[list[np.float64]]]:
        return [], []

    def update(self, new_frame: U8Mat):
        self.stack.update(new_frame)
        # 冷启动阶段（滑窗未填满）`noise_sigma` 为 0，此时沿用固定的绝对阈值。
        if self.bi_cfg.adaptive_bi_thre and (self.stack.noise_sigma != 0):
            self.bi_threshold_float = self.std2thre(self.stack.noise_sigma)
            self.bi_threshold = round(self.bi_threshold_float)

    @property
    def noise_sigma(self) -> float:
        """当前噪声标准差（DN）。未启用噪声评估时为 0。"""
        return self.stack.noise_sigma if self.stack.est_noise else 0.0

    def visu(self):
        return super().visu()

    def calculate_dy_mask(self, act: U8Mat):
        # if "dynamic_mask" is applied, stack and mask dst
        self.dy_mask_list.update(act)
        # TODO: 进一步使Dy_mask稳定作用在持续产生响应的区域，并提供可调整的阈值。
        np.less_equal(self.dy_mask_list.sum,
                      (self.dy_mask_list.length - 1) * 255,
                      out=self._dy_mask_buffer,
                      casting="unsafe")
        cv2.erode(self._dy_mask_buffer, self.cv_op, dst=self._dy_mask_buffer)
        # The history window has already copied act; the caller no longer needs
        # its unmasked contents, so reuse it for the result.
        cv2.multiply(act, self._dy_mask_buffer, dst=act)
        return act


class ClassicDetector(LineDetector):
    '''[uzanka-based detector](https://github.com/uzanka/MeteorDetector), in python.
    inspired by uzanka(https://github.com/uzanka)
    '''
    classic_max_size = 4

    def __init__(self, window_sec: float, fps: float, mask: U8Mat,
                 cfg: BinaryCfg, logger: BaseMetLog):
        # 4帧窗口（硬编码）
        window_sec = self.classic_max_size / fps
        super().__init__(window_sec, fps, mask, cfg, logger)

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
        linesp = cv2.HoughLinesP(dst,
                                 rho=1,
                                 theta=PI,
                                 threshold=self.hough_cfg.threshold,
                                 minLineLength=self.hough_cfg.min_len,
                                 maxLineGap=self.hough_cfg.max_gap)
        if linesp is None:
            linesp = np.array([])
        elif np.array(linesp).ndim == 3:
            linesp = linesp[:, 0, :]
        self.linesp: Sequence[list[int]] = linesp
        # TODO:
        # 1. Classic Detector的可视化接口尚未实现
        # 2. ClassicDetector的输出统一为METEOR判定。可能需要考虑逻辑是否会变更。
        cls_pred = np.zeros((len(self.linesp), self.num_cls))
        cls_pred[:, 0] = 1
        return self.linesp, cls_pred

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
                 cfg: BinaryCfg, logger: BaseMetLog):
        super().__init__(window_sec, fps, mask, cfg, logger)
        self._mean_buffer = np.empty(self.mask.shape, dtype=np.uint8)
        self._diff_buffer = np.empty(self.mask.shape, dtype=np.uint8)

    def detect(self):
        # Preprocessing
        # Mainly calculate diff_img (which basically equals to max-mid)
        light_img = self.stack.max
        np.subtract(light_img, self.stack.mean_into(self._mean_buffer),
                    out=self._diff_buffer)
        diff_img = cv2.medianBlur(self._diff_buffer, 3)

        # Post-processing后处理：二值化 + 闭运算
        _, dst = cv2.threshold(diff_img, self.bi_threshold, 255,
                               cv2.THRESH_BINARY)
        #dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, self.cv_op)
        dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, self.cv_op)

        # dynamic_mask 机制
        # if "dynamic_mask" is applied, stack and mask dst
        if self.dynamic_cfg.dy_mask:
            dst = self.calculate_dy_mask(dst)

        self.dst_sum = cast(float, cv2.countNonZero(dst) / self.mask_area * 100)
        gap = max(
            0, 1 - self.dst_sum / self.max_allow_gap) * self.hough_cfg.max_gap

        # 核心步骤：直线检测
        linesp = cv2.HoughLinesP(dst,
                                 rho=1,
                                 theta=PI,
                                 threshold=self.hough_cfg.threshold,
                                 minLineLength=self.hough_cfg.min_len,
                                 maxLineGap=gap)
        if linesp is None:
            linesp = np.array([])
        elif np.array(linesp).ndim == 3:
            linesp = linesp[:, 0, :]
        linesp = np.array([]) if linesp is None else linesp

        # 如果产生的响应数目非常多，忽略该帧
        # TODO: 会造成无法响应面积式的现象。需要调整。
        # 下调了阈值以提升面积召回。更合理的版本：
        self.lines_num = len(linesp)
        if self.lines_num > NUM_LINES_TOOMUCH:
            linesp = np.array([])

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
        self.linesp = linesp
        self.dst = dst
        # NMS
        # Another TODO: 不确定是否会产生额外的性能开销。
        if len(linesp) > 0:
            linesp, nonline_probs = lineset_nms(linesp)
            self.filtered_line_num = len(linesp)
            cls_pred = np.zeros((self.filtered_line_num, self.num_cls))
            others_col = get_name2id()["OTHERS"]
            cls_pred[:, others_col] = nonline_probs
            cls_pred[:, 0] = 1 - nonline_probs
        else:
            self.filtered_line_num = 0
            cls_pred = np.zeros((0, self.num_cls))
        return linesp, cls_pred

    def visu(self):
        """ 返回可视化时的所需实际参数。
        """
        x1, y1, x2, y2 = [-1] * 4
        if getattr(self.stack, "std_roi", None):
            x1, y1, x2, y2 = self.stack.std_roi
        visu_list: list[BaseVisuAttrs] = [
            ImgVisuAttrs("mix_bg",
                         img=self.dst // 255,
                         weight=0.5,
                         color="yellow"),
            TextVisu(
                "noise_sigma",
                position="left-top",
                color="green",
                text_list=[
                    TextColorPair(text=f"STD:{self.noise_sigma:.4f}")
                ]),
            TextVisu(
                "bi_value",
                position="left-top",
                color="green",
                text_list=[
                    TextColorPair(
                        text=
                        f"Bi_Threshold: {self.bi_threshold} (rounded from {self.bi_threshold_float:.4f})"
                    )
                ]),
            TextVisu(
                "lines_num",
                position="left-top",
                color="green",
                text_list=[
                    TextColorPair(
                        text=
                        f"Line num: {self.lines_num} (filtered: {self.filtered_line_num})"
                    )
                ]),
            TextVisu("area_ratio",
                     position="left-top",
                     color="green",
                     text_list=[
                         TextColorPair(text=f"Diff Area: {self.dst_sum:.2f}%")
                     ]),
            TextVisu("lines_warning",
                     position="left-top",
                     color="red",
                     text_list=[
                         TextColorPair(text="WARNING: TOO MANY LINES!" if self.
                                       lines_num > 10 else "")
                     ]),
            DrawRectVisu(
                "std_roi_area",
                pair_list=[SquareColorPair(dot_pair=([y1, x1], [y2, x2]))],
                color="purple")
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
                 cfg: BinaryCfg, logger: BaseMetLog):
        self.logger = logger
        self.logger.info(f"Momentum={(1 - 1 / (window_sec * fps)):.4f}")
        self.bg_maintainer = Uint8EMA(momentum=(1 - 1 / (window_sec * fps)))

    def update(self, new_frame: U8Mat) -> None:
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
        self.diff_img = ((self.cur_frame.astype(np.float64) + 100)
                         > self.bg_maintainer.cur_value).astype(np.uint8) * 255
        #self.diff_img[neg_value_mask] = 0
        self.post_update()
        return [], []

    def visu(self):
        ret: list[BaseVisuAttrs] = [
            ImgVisuAttrs("diff_mask",
                         img=self.diff_img,
                         color="yellow",
                         weight=0.5),
            TextVisu(
                "cur_emo_value",
                position="left-top",
                color="green",
                text_list=[
                    TextColorPair(
                        text=
                        f"Diff+: {np.mean(self.cur_frame.astype(np.float64) - self.bg_maintainer.cur_value):.4f}"
                    )
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
                 cfg: DLCfg, logger: BaseMetLog):
        # stack
        self.mask = mask
        self.logger = logger
        self.stack_maxsize = int(window_sec * fps)
        self.stack = SlidingWindow(n=self.stack_maxsize,
                                   size=self.mask.shape,
                                   dtype=np.uint8,
                                   force_int=True)
        self.model = init_model(cfg.model, logger=self.logger)

    def update(self, new_frame: U8Mat) -> None:
        self.stack.update(new_frame)

    def detect(self):
        self.result_pos, self.result_cls = self.model.forward(self.stack.max)
        if len(self.result_pos) == 0:
            return [], []
        # 调整反斜向流星xy坐标顺序
        for i, result_list in enumerate(self.result_pos):
            roi_img = crop_with_box(self.stack.max, Box.from_list(result_list))
            gradient_drct = calc_roi_gradient(roi_img)
            if np.isfinite(gradient_drct) and (int(gradient_drct //
                                                   (np.pi / 2)) % 2 == 1):
                self.result_pos[i, [1, 3]] = self.result_pos[i, [3, 1]]
        return self.result_pos, expand_cls_pred(self.result_cls)

    def visu(self):
        """ 返回可视化时的所需实际参数。

        Args:
            bg (np.array): 背景图像
            light (np.array): 响应图像
        """
        visu_list: list[BaseVisuAttrs] = [
            DrawRectVisu("results",
                         color="orange",
                         pair_list=[
                             SquareColorPair(dot_pair=([x[0], x[1]],
                                                       [x[2], x[3]]))
                             for x in self.result_pos
                         ])
        ]
        return visu_list


class BrightnessDetector(BaseDetector):
    """基于分区亮度跟踪的突变事件检测器。

    将图像划分为 R×C 的网格，对每个单元格维护亮度均值的 EMA 基线和方差 EMA。
    每帧计算各单元格当前亮度与基线的偏差，通过 z-score + 绝对阈值双重条件判定异常，
    再将触发的相邻单元格合并为 bounding box 输出。

    主要检测目标：闪电、火流星、人造天体再入等大面积亮度突变事件。

    Args:
        window_sec (float): 检测器窗口大小（秒），用于确定 EMA warmup 速度。
        fps (float): 等效帧率。
        mask (U8Mat): 有效区域掩模。
        cfg (BrightnessCfg): 亮度检测器配置。
        logger (BaseMetLog): 日志器。
    """

    def __init__(self, window_sec: float, fps: float, mask: U8Mat,
                 cfg: BrightnessCfg, logger: BaseMetLog):
        self.num_cls = get_num_class()
        self.logger = logger
        self.b_cfg = cfg.brightness
        self._brightness_event_id = get_name2id()["BRIGHTNESS_EVENT"]

        R, C = self.b_cfg.grid_rows, self.b_cfg.grid_cols
        H, W = mask.shape[:2]

        # 网格参数：截断到能被网格整除的尺寸
        self.cell_h = H // R
        self.cell_w = W // C
        self.grid_shape = (R, C)
        self.frame_size = (H, W)
        self.trimmed_h = R * self.cell_h
        self.trimmed_w = C * self.cell_w

        # 计算每个单元格的 mask 有效像素比例，低于阈值则标记为无效
        trimmed_mask = mask[:self.trimmed_h, :self.trimmed_w]
        cell_mask_means = trimmed_mask.reshape(R, self.cell_h, C,
                                               self.cell_w).mean(axis=(1, 3))
        self.valid_cells = (cell_mask_means / 255.0
                            > self.b_cfg.min_cell_valid)

        # EMA 维护基线亮度和方差，复用已有的 EMA 组件
        warmup_speed = max(int(window_sec * fps), 2)
        self.baseline_ema = EMA(momentum=self.b_cfg.ema_momentum,
                                warmup_speed=warmup_speed)
        self.var_ema = EMA(momentum=self.b_cfg.ema_momentum,
                           warmup_speed=warmup_speed)
        # 手动设定方差初始值为 1，避免早期除零
        self.var_ema.cur_value = np.ones((R, C), dtype=np.float32)

        self.t = 0

        # 可视化状态缓存
        self.triggered_grid = np.zeros((R, C), dtype=np.uint8)
        self.delta: Optional[np.ndarray] = None
        self.z_scores: Optional[np.ndarray] = None

        self.logger.info(
            f"BrightnessDetector initialized: grid={R}x{C}, "
            f"cell_size={self.cell_h}x{self.cell_w}, "
            f"warmup_speed={warmup_speed}, "
            f"valid_cells={int(np.sum(self.valid_cells))}/{R * C}")

    def update(self, new_frame: U8Mat) -> None:
        if len(new_frame.shape) == 3:
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        self.cur_frame = new_frame

    def detect(self) -> tuple[list[list[int]], list[list[np.float64]]]:
        self.t += 1
        cell_means = self._compute_cell_means(self.cur_frame)
        baseline = self.baseline_ema.cur_value

        # 计算偏差和 z-score
        delta = cell_means - baseline
        var = self.var_ema.cur_value
        z_scores = delta / np.sqrt(var + 1e-6)

        # 双阈值判定：z-score 阈值（防噪声环境漏报）+ 绝对值阈值（防静场景误报）
        triggered = ((z_scores > self.b_cfg.z_threshold)
                     & (delta > self.b_cfg.abs_threshold) & self.valid_cells)

        # 后置更新 EMA（在检测之后更新，防止突变帧自消除）
        self.baseline_ema.update(cell_means)
        self.var_ema.update(delta**2)

        # 更新可视化缓存
        self.triggered_grid = triggered.astype(np.uint8)
        self.z_scores = z_scores
        self.delta = delta

        if not triggered.any():
            return [], []

        # 合并触发区域为 bounding boxes
        boxes, scores = self._merge_to_boxes(triggered, z_scores)

        # 构建 cls_pred: 将置信度放在 BRIGHTNESS_EVENT 类别上
        cls_pred = np.zeros((len(boxes), self.num_cls))
        for i, score in enumerate(scores):
            cls_pred[i, self._brightness_event_id] = score

        return boxes, cls_pred.tolist()

    def _compute_cell_means(self, frame: U8Mat) -> np.ndarray:
        """通过 reshape 向量化计算每个网格单元格的平均亮度。

        将帧截断到能被网格整除的尺寸后，reshape 为 (R, cell_h, C, cell_w)，
        沿 cell 内维度求均值，一步得到 (R, C) 的结果。

        Args:
            frame: 灰度帧 (H, W)

        Returns:
            (R, C) 的 float32 数组
        """
        R, C = self.grid_shape
        trimmed = frame[:self.trimmed_h, :self.trimmed_w]
        return trimmed.reshape(R, self.cell_h, C,
                               self.cell_w).mean(axis=(1,
                                                       3)).astype(np.float32)

    def _merge_to_boxes(
            self, triggered: np.ndarray,
            z_scores: np.ndarray) -> tuple[list[list[int]], list[float]]:
        """将触发的网格单元格合并为 bounding boxes。

        当触发比例超过 global_ratio 时，判定为全局事件，输出全帧 box。
        否则使用连通域分析将相邻触发单元格合并为独立的 box。

        Args:
            triggered: (R, C) bool 数组，标记触发的单元格
            z_scores: (R, C) float 数组，各单元格的 z-score

        Returns:
            (boxes, scores) — boxes 为 [x1,y1,x2,y2] 列表，scores 为置信度列表
        """
        H, W = self.frame_size
        num_valid = max(int(np.sum(self.valid_cells)), 1)
        trigger_ratio = np.sum(triggered) / num_valid

        # 全局事件：超过 global_ratio 比例的有效单元格触发
        if trigger_ratio > self.b_cfg.global_ratio:
            max_z = float(np.max(z_scores[triggered]))
            score = self._z_to_score(max_z)
            return [[0, 0, W, H]], [score]

        # 局部事件：使用连通域分析合并相邻触发单元格
        grid_img = triggered.astype(np.uint8) * 255
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            grid_img, connectivity=4)

        boxes: list[list[int]] = []
        scores: list[float] = []
        for label_id in range(1, num_labels):
            component_mask = (labels == label_id)
            rows, cols = np.where(component_mask)
            r_min, r_max = int(rows.min()), int(rows.max())
            c_min, c_max = int(cols.min()), int(cols.max())
            # 网格坐标转换为像素坐标
            x1 = c_min * self.cell_w
            y1 = r_min * self.cell_h
            x2 = min((c_max + 1) * self.cell_w, W)
            y2 = min((r_max + 1) * self.cell_h, H)
            boxes.append([x1, y1, x2, y2])
            max_z = float(np.max(z_scores[component_mask]))
            scores.append(self._z_to_score(max_z))

        return boxes, scores

    def _z_to_score(self, z: float) -> float:
        """将 z-score 通过 sigmoid 映射为 (0.5, 1.0) 区间的置信分数。"""
        return float(1.0 / (1.0 + np.exp(-(z - self.b_cfg.z_threshold))))

    def visu(self) -> list[BaseVisuAttrs]:
        R, C = self.grid_shape
        visu_list: list[BaseVisuAttrs] = []

        # 绘制触发的网格单元格
        if self.triggered_grid.any():
            pairs: list[SquareColorPair] = []
            for r in range(R):
                for c in range(C):
                    if self.triggered_grid[r, c]:
                        y1 = r * self.cell_h
                        x1 = c * self.cell_w
                        y2 = min((r + 1) * self.cell_h, self.frame_size[0])
                        x2 = min((c + 1) * self.cell_w, self.frame_size[1])
                        pairs.append(
                            SquareColorPair(dot_pair=([x1, y1], [x2, y2])))
            if pairs:
                visu_list.append(
                    DrawRectVisu("brightness_trigger",
                                 pair_list=pairs,
                                 color="red"))

        # 文字信息
        if self.delta is not None:
            mean_delta = float(np.mean(self.delta[self.valid_cells]))
            triggered_n = int(np.sum(self.triggered_grid))
            total_valid = int(np.sum(self.valid_cells))
            visu_list.append(
                TextVisu("brightness_info",
                         position="left-top",
                         color="cyan",
                         text_list=[
                             TextColorPair(
                                 text=f"Brightness \u0394:{mean_delta:.1f} "
                                 f"Triggered:{triggered_n}/{total_valid}")
                         ]))

        return visu_list
