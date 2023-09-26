import cv2
import numpy as np
from typing import Callable, Any
from abc import abstractmethod, ABCMeta
from .utils import NumpySlidingWindow, generate_group_interpolate, EMA, PI, identity
r"""

Relation of Detectors in MetDetPy:

                               |--ClassicDetector
              |--LineDetector--|
BaseDetector--|                |--M3Detector
              |
              |----MLDetector-----YOLODetector
"""
NUM_LINES_TOOMUCH=500

class SNRSlidingWindow(NumpySlidingWindow):
    """
    ## 带有snr评估的滑窗管理器

    基于滑动窗口维护序列，并可以用于维护前景图像，参考背景图像，评估信噪比的模块。
    """

    def __init__(self, n, ref_mask, area=0, mul=2, noise_moment=0.99) -> None:
        super().__init__(n, ref_mask.shape, dtype=np.uint8)
        self.noise_ema = EMA(momentum=noise_moment)
        self.std_interval = mul * n
        self.est_std = self.select_subarea(ref_mask, area=area)

    def update(self, new_frame):
        super().update(new_frame)
        # 每std_interval时间更新一次std
        if self.timer % self.std_interval == 0:
            self.noise_ema.update(self.est_std(self.sliding_window -
                                               self.mean))

    def select_subarea(self, mask, area: float) -> Callable:
        """用于选择一个尽量好的子区域评估STD。

        Args:
            mask (_type_): _description_
            area (float, optional): _description_. Defaults to 0.1.

        Returns:
            _type_: _description_
        """
        h, w, c = mask.shape
        if area == 0:
            self.std_roi = (h // 2, w // 2, 0, 0)
            return lambda *args: 0

        sub_rate = area**(1 / 2)
        sub_h, sub_w = int(h * sub_rate), int(w * sub_rate)
        x1, y1 = 0, (w - sub_w) // 2
        area = (sub_h - 1) * (sub_w - 1)
        light_ratio = np.sum(mask[x1:x1 + sub_h, y1:y1 + sub_w]) / area
        while light_ratio < 1:
            x1 += 10
            new_ratio = np.sum(mask[x1:x1 + sub_h, y1:y1 + sub_w]) / area
            if new_ratio < light_ratio or y1 + sub_w > w:
                x1 -= 10
                break
            light_ratio = new_ratio
        self.std_roi = (x1, y1, x1 + sub_h, y1 + sub_w)
        return lambda imgs: np.std(imgs[:, x1:x1 + sub_h, y1:y1 + sub_w])

    @property
    def std(self):
        return self.noise_ema.cur_value


class BaseDetector(metaclass=ABCMeta):
    """An abstract base class for detector.
    
    To implement a detector, you should:xxxxxx 
    TODO: add more explaination.

    VideoLoader class handles the whole video-to-frame process, which should include: 
    1. init - of course, initialize your detector status.
    2. update - update a frame to your detector. your detector may create a stack to store them... or not.
    3. detect - the kernel function that receive no argument and return candidate meteor position series.
    4. visu_closure - a optional closure drawing function that draw debug information on the input image.

    """

    @abstractmethod
    def __init__(self, *args) -> None:
        pass

    @abstractmethod
    def update(self, new_frame) -> None:
        pass

    @abstractmethod
    def detect(self) -> np.ndarray:
        pass

    def visu_closure(self, bg: np.ndarray, extra_info=[]) -> Callable:
        return identity


class LineDetector(BaseDetector):
    """基于"二值化-Hough直线检测"的检测器类。

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
    sensitivity_func = {
        "low": lambda x: 1.5 * x**2 + 4.2,
        "normal": lambda x: 1.2 * x**2 + 3.6,
        "high": lambda x: 0.9 * x**2 + 3,
    }
    abs_sensitivity = {"high": 3, "normal": 5, "low": 7, "very_low": 10}
    cv_op = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def __init__(self, window_sec, fps, mask, bi_cfg, hough_cfg, dynamic_cfg):
        # mask
        self.mask = mask
        self.mask_area = np.sum(self.mask)
        # cfg
        self.bi_cfg = bi_cfg
        self.hough_cfg = hough_cfg
        self.dynamic_cfg = dynamic_cfg
        # stack
        self.stack_maxsize = int(window_sec * fps)
        self.stack = SNRSlidingWindow(n=self.stack_maxsize,
                                      ref_mask=mask,
                                      area=self.bi_cfg.area)
        # 加载自适应阈值的配置参数
        if self.bi_cfg.adaptive_bi_thre:
            self.std2thre = self.sensitivity_func[self.bi_cfg.sensitivity]
            self.bi_threshold = self.abs_sensitivity[self.bi_cfg.sensitivity]

        # 如果启用动态蒙版（dynamic mask），在此处构建另一个滑窗管理
        if self.dynamic_cfg.dy_mask:
            self.dy_mask_list = SNRSlidingWindow(n=self.stack_maxsize,
                                                 ref_mask=mask,
                                                 area=self.bi_cfg.area)

        # 动态间隔()
        if self.dynamic_cfg.dy_gap:
            self.max_allow_gap = self.dynamic_cfg.dy_gap
            self.fill_thre = self.dynamic_cfg.fill_thre

    def detect(self) -> tuple[list, Callable]:
        return [], identity

    def update(self, new_frame):
        self.stack.update(new_frame)
        if self.bi_cfg.adaptive_bi_thre and (self.stack.std != 0):
            self.bi_threshold = self.std2thre(self.stack.std)

    def visu_closure(self,
                     bg: np.ndarray,
                     extra_info=[]) -> Callable[..., Any]:
        return super().visu_closure(bg, extra_info)

    def calculate_dy_mask(self, act):
        # if "dynamic_mask" is applied, stack and mask dst
        self.dy_mask_list.update(act)
        # TODO: 你要不要看看你在写什么.jpg
        dy_mask = cv2.threshold(
            np.sum(self.dy_mask_list.sliding_window // 255,
                   axis=0,
                   dtype=np.uint8), self.dy_mask_list.length - 1, 1,
            cv2.THRESH_BINARY_INV)[-1]
        dy_mask = cv2.erode(dy_mask, self.cv_op)
        # TODO: 加入通道维之后需要修正
        if len(dy_mask.shape) == 2: dy_mask = dy_mask[:, :, None]
        return np.multiply(act, dy_mask)


class ClassicDetector(LineDetector):
    '''[uzanka-based detector](https://github.com/uzanka/MeteorDetector), in python.
    inspired by uzanka(https://github.com/uzanka)
    '''
    classic_max_size = 4

    def __init__(self, window_sec, fps, mask, bi_cfg, hough_cfg, dynamic_cfg):
        # 4帧窗口（硬编码）
        window_sec = self.classic_max_size / fps
        super().__init__(window_sec, fps, mask, bi_cfg, hough_cfg, dynamic_cfg)

    def detect(self):
        id3, id2, id1, id0 = [
            self.stack.cur_index - i for i in range(self.classic_max_size)
        ]
        sw = self.stack.sliding_window
        # 短于4帧时不进行判定
        if self.stack.timer < self.stack_maxsize:
            return [], sw[id3]

        # 差分2,3帧，二值化，膨胀（高亮为有差异部分）
        diff23 = cv2.absdiff(sw[id2], sw[id3])
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

        self.linesp = [] if self.linesp is None else self.linesp[0]
        return self.linesp, self.visu_closure(sw[id3], dst)


class M3Detector(LineDetector):
    """Detector, but version spring.

    主要工作原理： 以X帧为窗口的帧差法 （最大值-中值）。
    
    20221109：Update：利用更大的滑窗范围（滑窗中值法）和简化的最大值-阈值方法提升精度与速度
    
    采取了相比原算法更激进的阈值和直线检测限，将假阳性样本通过排除记录流星的算法移除。

    Args:
        stack (_type_): _description_
        threshold (_type_): _description_
        drawing (_type_): _description_

    Returns:
        _type_: _description_
    """

    def detect(self) -> tuple:
        # Preprocessing
        # Mainly calculate diff_img (which basically equals to max-mid)
        light_img = self.stack.max
        diff_img = (light_img - self.stack.mean).astype(dtype=np.uint8)
        diff_img = cv2.medianBlur(diff_img, 3)

        # Post-processing后处理
        _, dst = cv2.threshold(diff_img, self.bi_threshold, 255,
                               cv2.THRESH_BINARY)
        dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, self.cv_op)

        # TODO: 加入通道维之后需要修正dst
        if len(dst.shape) == 2: dst = dst[:, :, None]

        # dynamic_mask 机制
        # if "dynamic_mask" is applied, stack and mask dst
        if self.dynamic_cfg.dy_mask:
            dst = self.calculate_dy_mask(dst)

        # dynamic_gap机制
        # 根据产生的响应比例适量减少gap
        # 一定程度上能够改善对低信噪比场景的误检
        dst_sum = np.sum(dst / 255.) / self.mask_area * 100  # type: ignore
        gap = max(0, 1 - dst_sum / self.max_allow_gap) * self.hough_cfg.max_gap

        # 核心步骤：直线检测
        linesp = cv2.HoughLinesP(dst,
                                 rho=1,
                                 theta=PI,
                                 threshold=self.hough_cfg.threshold,
                                 minLineLength=self.hough_cfg.min_len,
                                 maxLineGap=gap)
        linesp = np.array([]) if linesp is None else linesp[:, 0, :]

        # 如果产生的响应数目非常多，忽略该帧
        lines_num = len(linesp)
        if lines_num > NUM_LINES_TOOMUCH:
            linesp = np.array([])

        # 后处理：对于直线进行质量评定，过滤掉中空比例较大的直线
        # 这一步骤会造成一些暗弱流星的丢失。
        if len(linesp) > 0:
            line_pts = generate_group_interpolate(linesp)
            line_score = np.array([
                np.sum(dst[line_pt[1], line_pt[0]]) / (len(line_pt[0]) * 255)
                for line_pt in line_pts
            ])
            #for line, line_pt in zip(linesp,line_pts):
            #    print(line, line_pt)
            #    print(dst[line_pt[1], line_pt[0]])
            linesp = linesp[line_score > self.fill_thre]
            lines_num = len(linesp)

        texts = [(f"Line num: {lines_num}", (0, 255, 0)),
                 (f"Diff Area: {dst_sum:.2f}%", (0, 255, 0))]
        if lines_num > 10:
            texts.append(
                ("WARNING: TOO MANY LINES!", (0, 0, 255)))  # type: ignore
        return linesp, self.visu_closure(light_img, dst, extra_info=texts)

    def visu_closure(self, bg, light, extra_info=[]):
        """ 简单实现的闭包式的绘图API，用于支持可视化

        Args:
            bg (np.array): 背景图像
            light (np.array): 响应图像
        """

        def core_drawer():
            # TODO: 兼容3色/单色两种场景
            p = cv2.cvtColor(light, cv2.COLOR_GRAY2RGB)
            b = cv2.cvtColor(bg, cv2.COLOR_GRAY2RGB)
            p[:, :, 0] = 0
            cb_img = cv2.addWeighted(b, 1, p, 0.5, 1)
            text_h = 20
            if getattr(self, "ref_ema", None):
                x1, y1, x2, y2 = self.stack.std_roi
                cb_img = cv2.rectangle(cb_img, (y1, x1), (y2, x2),
                                       color=(128, 64, 128),
                                       thickness=2)
                cb_img = cv2.putText(cb_img, f"STD:{self.stack.std:.4f};",
                                     (10, text_h), cv2.FONT_HERSHEY_COMPLEX,
                                     0.5, (0, 255, 0), 1)
                text_h += 20
                cb_img = cv2.putText(cb_img,
                                     f"Bi_Threshold: {self.bi_threshold:.2f}",
                                     (10, text_h), cv2.FONT_HERSHEY_COMPLEX,
                                     0.5, (0, 255, 0), 1)
                text_h += 20
            for (text, color) in extra_info:
                cb_img = cv2.putText(cb_img, text, (10, text_h),
                                     cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
                text_h += 20
            return cb_img

        return core_drawer