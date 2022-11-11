import cv2
import numpy as np

from .utils import m3func

pi = np.pi / 180.0


def init_detector(name, detect_cfg, debug_mode, fps):
    if name == "ClassicDetector":
        return ClassicDetector(-1, detect_cfg, debug_mode=debug_mode)

    elif name == "M3Detector":
        # Odd Length for M3Detector
        window_size = max(int(detect_cfg["window_sec"] * fps), 2)
        if window_size % 2 == 0:
            window_size += 1
        return M3Detector(window_size, detect_cfg, debug_mode=debug_mode)


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
            debug_mode (bool, optional): 是否启用调试模式。调试模式将打印更多检测细节. Defaults to False.
    

    self.stack用于放置检测窗口的帧图像。
    self.linesp用于放置检测结果。(对于基于直线检测的方法)

    """

    def __init__(self, stack_maxsize, cfg, debug_mode=False):
        self.stack_maxsize = stack_maxsize
        self.cfg_loader(cfg)
        self.debug_mode = debug_mode
        self.stack = []
        pass

    def detect(self):
        pass

    def update(self, new_frames):
        self.stack.append(new_frames)
        self.stack = self.stack[-self.stack_maxsize:]

    def draw_light_on_bg(self, bg, light):
        """ 简单实现的闭包式的绘图API，用于支持可视化响应等

        Args:
            bg (np.array): 背景图像
            light (np.array): 响应图像
        """

        def core_drawer():
            p = cv2.cvtColor(light, cv2.COLOR_GRAY2RGB)
            b = cv2.cvtColor(bg, cv2.COLOR_GRAY2RGB)
            p[:, :, [0, 1]] = 0
            return cv2.addWeighted(b, 1, p, 1, 1)

        return core_drawer

    def cfg_loader(self, cfg: dict):
        for name, value in cfg.items():
            setattr(self, name, value)


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
        debug_mode (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    def __init__(self, *args, **kwargs):
        # 必须包含的参数
        # bi_threshold line_threshold self.max_gap img_shape
        # 需要可视化时包含的参数
        # mask
        super().__init__(*args, **kwargs)
        self.ref_img = np.zeros(self.img_shape, dtype=float)

    def update(self, new_frames):
        # 更新背景估计参考
        # TODO: 之后考虑添加阈值估计参考（即方差）
        if len(self.stack) >= self.stack_maxsize:
            #p = (self.stack - self.ref_img / len(self.stack))
            # 方差估算可以左右SNR的参考依据。但总的来说低于单帧之间。
            #print(np.mean(p),np.std(p))
            self.ref_img -= self.stack[0]
        self.ref_img += new_frames

        super().update(new_frames)

    def detect(self) -> tuple:
        light_img = np.max(self.stack, axis=0)
        diff_img = (light_img -
                    (self.ref_img / len(self.stack))).astype(dtype=np.uint8)

        diff_img = cv2.medianBlur(diff_img, 3)
        _, dst = cv2.threshold(diff_img, self.bi_threshold, 255,
                               cv2.THRESH_BINARY)
        dst = cv2.medianBlur(dst, 3)
        # TODO: 这一套对噪点大的不太适用。。。
        #dst = cv2.morphologyEx(
        #    dst, cv2.MORPH_OPEN,
        #    cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
        dst = cv2.morphologyEx(
            dst, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        dst = cv2.dilate(
            dst,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        )
        linesp = cv2.HoughLinesP(dst, 0.8, pi, self.hough_threshold,
                                 self.min_len, self.max_gap)

        linesp = [] if linesp is None else linesp[0]
        return linesp, self.draw_light_on_bg(light_img, dst)

    def draw_on(self, canvas):
        if self.debug_mode:
            diff_img = cv2.cvtColor(np.array(diff_img, np.uint8),
                                    cv2.COLOR_GRAY2BGR)
            drawing = np.repeat(np.expand_dims(drawing, axis=-1), 3, axis=-1)
            y, x = self.visual_param
            canvas = np.zeros((x * 2, y * 2, 3), dtype=np.uint8)
            canvas[:x, :y] = cv2.resize(drawing, (y, x))
            canvas[:x, y:] = cv2.resize(diff_img, (y, x))
            canvas[x:, :y] = cv2.cvtColor(
                cv2.resize(
                    DrawHist(diff_img, self.mask, threshold=self.bi_threshold),
                    (y, x)), cv2.COLOR_GRAY2BGR)
            canvas[x:, y:] = cv2.resize(drawing, (y, x))
            drawing = canvas
