import cv2
import numpy as np

from .utils import m3func, RefEMA

pi = np.pi / 180.0

# version I
sensi_func = {
    "low": lambda x: 1.5 * x**2 + 4.2,
    "normal": lambda x: 1.2 * x**2 + 3.6,
    "high": lambda x: 0.9 * x**2 + 3,
}

absolute_sensitivity_mapping = {
    "high": 3,
    "normal": 5,
    "low": 7,
    "very_low": 10
}


def init_detector(name, detect_cfg, fps):
    if name == "ClassicDetector":
        return ClassicDetector(-1, detect_cfg)

    elif name == "M3Detector":
        window_size = int(detect_cfg["window_sec"] * fps)
        return M3Detector(window_size, detect_cfg)


def DrawHist(src, mask, hist_num=256, threshold=0):
    """使用Opencv给给定的图像上绘制直方图。绘制的直方图大小和输入图像尺寸相当。

    Args:
        src (_type_): _description_
        mask (_type_): _description_
        hist_num (int, optional): _description_. Defaults to 256.
        threshold (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([src], [0], mask, [hist_num],
                        [0, hist_num - 1])[:, 0] / np.sum(mask)
    h, w = src.shape
    bw = w / hist_num
    canvas = np.zeros_like(src, dtype=np.uint8)
    i = threshold
    canvas = cv2.rectangle(canvas, (int(bw * i), h), (int(bw * i), 0),
                           [255, 255, 255], 2, -1)
    for i, bar in enumerate(hist):
        canvas = cv2.rectangle(canvas, (int(bw * i), h),
                               (int(bw * (i + 1)), h - int(bar * h)),
                               [128, 128, 128], 2, -1)
    return canvas


class BaseDetector(object):
    """检测器类。

    Args:
            stack_size (_type_): 检测器的窗口大小
            cfg (_type_): 参数配置。
    

    self.stack用于放置检测窗口的帧图像。
    self.linesp用于放置检测结果。(对于基于直线检测的方法)

    """

    def __init__(self, stack_maxsize, cfg):
        self.stack_maxsize = stack_maxsize
        self.stack = []
        # load all cfg to self.attributes
        for name, value in cfg.items():
            setattr(self, name, value)

        # default version does not support ada_threshold
        self.bi_threshold = self.bi_cfg["init_value"]

    def detect(self):
        pass

    def update(self, new_frame):
        self.stack.append(new_frame)
        self.stack = self.stack[-self.stack_maxsize:]

    def draw_light_on_bg(self, bg, light, text=None):
        """ 简单实现的闭包式的绘图API，用于支持可视化

        Args:
            bg (np.array): 背景图像
            light (np.array): 响应图像
        """

        def core_drawer():
            p = cv2.cvtColor(light, cv2.COLOR_GRAY2RGB)
            b = cv2.cvtColor(bg, cv2.COLOR_GRAY2RGB)
            p[:, :, 0] = 0
            cb_img = cv2.addWeighted(b, 1, p, 0.5, 1)
            if getattr(self, "ref_ema", None):
                if getattr(self.ref_ema, "roi", None):
                    x1, y1, x2, y2 = self.ref_ema.roi
                    cb_img = cv2.rectangle(cb_img, (y1, x1), (y2, x2),
                                           color=(128, 64, 128),
                                           thickness=2)
                cb_img = cv2.putText(
                    cb_img,
                    f"STD:{self.ref_ema.mean:.4f}; (Real noise: {self.ref_ema.noise:.4f})",
                    (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
                cb_img = cv2.putText(cb_img,
                                     f"Bi_Threshold: {self.bi_threshold:.2f}",
                                     (10, 40), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                     (0, 255, 0), 1)

            return cb_img

        return core_drawer


class ClassicDetector(BaseDetector):
    '''基于日本人版本实现的检测器类别。'''

    # 必须包含的参数
    # bi_threshold line_threshold self.max_gap
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 4帧窗口（硬编码）
        self.stack_maxsize = 4

    def detect(self):
        # 短于4帧时不进行判定
        if len(self.stack) < self.stack_maxsize:
            return False, [], self.stack[-1]
        # 差分2,3帧，二值化，膨胀（高亮为有差异部分）
        diff23 = cv2.absdiff(self.stack[2], self.stack[3])
        _, diff23 = cv2.threshold(diff23, self.bi_threshold, 255,
                                  cv2.THRESH_BINARY)
        diff23 = cv2.dilate(
            diff23,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        )
        diff23 = 255 - diff23
        ## 用diff23和0,1帧做位与运算（掩模？），屏蔽2,3帧的差一部分
        f1 = cv2.bitwise_and(diff23, self.stack[0])
        f2 = cv2.bitwise_and(diff23, self.stack[1])
        ## 差分0,1帧，二值化，膨胀（高亮有差异部分）
        dst = cv2.absdiff(f1, f2)
        _, dst = cv2.threshold(dst, self.bi_threshold, 255, cv2.THRESH_BINARY)
        dst = cv2.dilate(
            dst,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        )
        # 对0,1帧直线检测（即：在屏蔽了2,3帧变化的图上直线检测。为毛？）
        # 所以即使检出应该也是第一帧上面检出。
        self.linesp = cv2.HoughLinesP(dst, 1, pi, 1, self.min_len,
                                      self.max_gap)

        linesp = [] if linesp is None else linesp[0]
        return linesp, self.draw_light_on_bg(self.stack[3], dst)


class M3Detector(BaseDetector):
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

    cv_op = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def __init__(self, *args, **kwargs):
        # 必须包含的参数
        # bi_threshold line_threshold self.max_gap mask
        super().__init__(*args, **kwargs)

        self.area = 0
        # 加载自适应阈值的配置参数
        if self.adaptive_bi_thre:
            self.sensitivity = self.bi_cfg["sensitivity"]
            self.std2thre = sensi_func[self.sensitivity]
            self.bi_threshold = absolute_sensitivity_mapping[self.sensitivity]
            self.area = self.bi_cfg["area"]
        # 使用RefEMA作为滑窗数值管理器
        self.ref_ema = RefEMA(n=self.stack_maxsize,
                              ref_mask=self.img_mask,
                              area=self.area)

    def update(self, new_frame):
        self.ref_ema.update(new_frame)
        if self.adaptive_bi_thre and (self.ref_ema.mean != np.inf):
            self.bi_threshold = self.std2thre(self.ref_ema.mean)

    def detect(self) -> tuple:
        light_img = self.ref_ema.li_img
        diff_img = (light_img - self.ref_ema.bg_img).astype(dtype=np.uint8)

        diff_img = cv2.medianBlur(diff_img, 3)
        _, dst = cv2.threshold(diff_img, self.bi_threshold, 255,
                               cv2.THRESH_BINARY)
        dst = cv2.medianBlur(dst, 3)
        dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, self.cv_op)
        dst = cv2.dilate(dst, self.cv_op)
        linesp = cv2.HoughLinesP(dst,
                                 rho=0.8,
                                 theta=pi,
                                 threshold=self.hough_threshold,
                                 minLineLength=self.min_len,
                                 maxLineGap=self.max_gap)

        linesp = [] if linesp is None else linesp[0]
        return linesp, self.draw_light_on_bg(
            light_img,
            dst,
        )
