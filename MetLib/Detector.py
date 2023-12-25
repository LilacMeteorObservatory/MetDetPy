"""

Relation of Detectors in MetDetPy:

                               |--ClassicDetector
              |--LineDetector--|
BaseDetector--|                |--M3Detector
              |
              |----MLDetector-----YOLODetector
"""

import cv2
import numpy as np
from typing import Callable, Any, Optional
from abc import abstractmethod, ABCMeta
from .utils import SlidingWindow, generate_group_interpolate, EMA, PI

NUM_LINES_TOOMUCH = 100


class SNR_SW(SlidingWindow):
    """
    ## 检测器使用的滑窗管理器。

    基于滑动窗口维护序列，可以用于产生维护前景图像，参考背景图像，评估信噪比。
    """

    def __init__(self,
                 n,
                 mask,
                 est_snr=True,
                 est_area=0,
                 noise_moment=0.99,
                 nz_interval=1) -> None:
        self.est_snr = est_snr
        self.nz_interval = nz_interval
        # 主滑窗，维护图像
        super().__init__(n,
                         size=mask.shape,
                         dtype=np.uint8,
                         force_int=True,
                         calc_std=False)
        # 需要评估信噪比时，额外的子滑窗
        if self.est_snr:
            self.noise_ema = EMA(momentum=noise_moment)
            self.std_interval = self.nz_interval * n
            # TODO: area=0 的处置还没写
            self.get_subarea = self.select_subarea(mask, area=est_area)
            sub_h, sub_w = self.std_roi[2] - self.std_roi[0], self.std_roi[
                3] - self.std_roi[1]
            self.sub_sw = SlidingWindow(n,
                                        size=(sub_h, sub_w),
                                        dtype=np.uint8,
                                        force_int=True,
                                        calc_std=True)

    def update(self, new_frame):
        super().update(new_frame)
        # 每std_interval时间更新一次std
        # 经验公式：当每隔std_interval计算一次时，结果应当除以sqrt(std_interval)以修正数据范围。
        if self.est_snr and (self.timer % self.std_interval == 0):
            self.subframe = self.get_subarea(new_frame)
            self.sub_sw.update(self.subframe)
            self.noise_ema.update(self.sub_sw.std / np.sqrt(self.std_interval))

    def select_subarea(self, mask, area: float) -> Callable:
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
            return lambda *args: 0

        sub_rate = area**(1 / 2)
        sub_h, sub_w = int(h * sub_rate), int(w * sub_rate)
        x1, y1 = (h - sub_h) // 2, (w - sub_w) // 2
        area = (sub_h - 1) * (sub_w - 1)
        light_ratio = np.sum(mask[x1:x1 + sub_h, y1:y1 + sub_w]) / area
        while light_ratio < 1:
            x1 += 10
            new_ratio = np.sum(mask[x1:x1 + sub_h, y1:y1 + sub_w]) / area
            if new_ratio < light_ratio or x1 + sub_h < h:
                x1 -= 10
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
    def __init__(self, *args) -> None:
        pass

    @abstractmethod
    def update(self, new_frame) -> None:
        pass

    @abstractmethod
    def detect(self) -> np.ndarray:
        pass

    def visu(self) -> dict:
        return {}


class LineDetector(BaseDetector):
    """基于"二值化-Hough直线检测"的检测器类。(作为抽象类，并不会产生检测结果)
    （迭代中，该版本并不稳定）

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
    sensitivity_func = {
        "low": lambda x: 1.5 * x**2 + 4.2,
        "normal": lambda x: 1.2 * x**2 + 3.6,
        "high": lambda x: 0.9 * x**2 + 3,
    }
    abs_sensitivity = {"high": 3, "normal": 5, "low": 7, "very_low": 10}
    cv_op = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def __init__(self, window_sec, fps, mask, bi_cfg, hough_cfg, dynamic_cfg):
        self.mask = mask
        self.mask_area = np.sum(self.mask)
        # cfg
        self.bi_cfg = bi_cfg
        self.hough_cfg = hough_cfg
        self.dynamic_cfg = dynamic_cfg
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

        # 如果启用动态蒙版（dynamic mask），在此处构建另一个滑窗管理
        if self.dynamic_cfg.dy_mask:
            self.dy_mask_list = SlidingWindow(n=self.stack_maxsize,
                                              size=self.mask.shape,
                                              dtype=np.uint8,
                                              force_int=True)

        # 动态间隔()
        if self.dynamic_cfg.dy_gap:
            self.max_allow_gap = self.dynamic_cfg.dy_gap
            self.fill_thre = self.dynamic_cfg.fill_thre

    def detect(self) -> tuple[list, dict[str, Any]]:
        return [], {}

    def update(self, new_frame: np.ndarray):
        self.stack.update(new_frame)
        if self.bi_cfg.adaptive_bi_thre and (self.stack.snr != 0):
            self.bi_threshold = self.std2thre(self.stack.snr)

    def visu(self) -> dict:
        return super().visu()

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
        return self.linesp, self.visu(sw[id3], dst)

    def visu(self, img, dst):
        raise NotImplementedError


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

        data_info = [[
            "text", "left-top", {
                "text": f"Line num: {lines_num}",
                "color": "green"
            }
        ],
                     [
                         "text", "left-top", {
                             "text": f"Diff Area: {dst_sum:.2f}%",
                             "color": "green"
                         }
                     ]]
        if lines_num > 10:
            data_info.append([
                "text", "left-top", {
                    "text": "WARNING: TOO MANY LINES!",
                    "color": "red"
                }
            ])

        return linesp, self.visu(light_img, dst, extra_info=data_info)

    def visu(self, bg, light, extra_info: Optional[list] = None) -> dict:
        """ 构造可视化时使用的

        Args:
            bg (np.array): 背景图像
            light (np.array): 响应图像
        """

        def core_drawer():
            """
            LineDetector只支持Monochrome，因此无需特别处置。
            """
            p = cv2.cvtColor(light, cv2.COLOR_GRAY2RGB)
            b = cv2.cvtColor(bg, cv2.COLOR_GRAY2RGB)
            p[:, :, 0] = 0
            cb_img = cv2.addWeighted(b, 1, p, 0.5, 1)
            return cb_img

        data_info = []
        # TODO: 似乎没有设置哪儿可以关闭这个。。
        #if getattr(self, "ref_ema", None):
        x1, y1, x2, y2 = self.stack.std_roi
        data_info.append(
            ["rectangle", [(y1, x1), (y2, x2)], {
                "color": "purple"
            }])
        data_info.append([
            "text", "left-top", {
                "text": f"STD:{self.stack.snr:.4f};",
                "color": "green"
            }
        ])
        data_info.append([
            "text", "left-top", {
                "text": f"Bi_Threshold: {self.bi_threshold:.2f}",
                "color": "green"
            }
        ])
        if extra_info:
            data_info.extend(extra_info)

        return {"bg": core_drawer, "info": data_info}

class MLDetector(BaseDetector):
    """基于机器学习模型管线的检测器。(作为抽象类，并不会产生检测结果)
    （迭代中，该版本并不稳定）

    可接受单通道/三通道的输入。

    Args:
        BaseDetector (_type_): _description_
    """

    """基于"二值化-Hough直线检测"的检测器类。
    
    # 可启用特性：
    暂无
    """

    def __init__(self, window_sec,mask, weight_path, fps) -> None:
        # stack
        self.mask = mask
        self.stack_maxsize = int(window_sec * fps)
        self.stack = SlidingWindow(n=self.stack_maxsize,
                            size=self.mask.shape,
                            dtype=np.uint8)
        # load model
        pass
                            
    def update(self, new_frame) -> None:
        self.stack.update(new_frame)

    def detect(self):
        pass

    def visu(self) -> dict:
        return {}