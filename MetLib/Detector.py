import cv2
import numpy as np

from .utils import m3func

pi = 3.141592653589793 / 180.0


def init_detector(name, detect_cfg, debug_mode, fps):
    if name == "ClassicDetector":
        return ClassicDetector(-1, detect_cfg, debug_mode)

    elif name == "M3Detector":
        # Odd Length for M3Detector
        window_size = max(int(detect_cfg["window_sec"] * fps), 2)
        if window_size % 2 == 0:
            window_size += 1
        if detect_cfg["median_sampling_num"] == -1:
            detect_cfg.update(median_skipping=1)
        else:
            assert detect_cfg[
                "median_sampling_num"] >= 3, "You must set median_sampling_num to 3 or larger."
            detect_cfg.update(median_skipping=(window_size - 1) //
                              (detect_cfg["median_sampling_num"] - 1))
        return M3Detector(window_size, detect_cfg, debug_mode)


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

    def draw_on(self, canvas):
        """
        用于在给定画布上绘制有关包含中间状态的检测细节。

        Args:
            canvas (_type_): _description_
        """
        pass

    def cfg_loader(self, cfg: dict):
        for name, value in cfg.items():
            setattr(self, name, value)


class ClassicDetector(BaseDetector):
    '''基于日本人版本实现的检测器类别。'''

    # 必须包含的参数
    # bi_threshold line_threshold self.line_minlen
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
        self.linesp = cv2.HoughLinesP(dst, 1, pi, self.line_threshold,
                                      self.line_minlen, 0)
        if self.linesp is None:
            return False, [], dst
        return True, self.linesp[0], dst


class M3Detector(BaseDetector):
    """Detector, but version spring.

    主要工作原理： 以X帧为窗口的帧差法 （最大值-中值）。
    
    采取了相比原算法更激进的阈值和直线检测限，将假阳性样本通过排除记录流星的算法移除。

    Args:
        stack (_type_): _description_
        threshold (_type_): _description_
        drawing (_type_): _description_
        debug_mode (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    # 必须包含的参数
    # bi_threshold line_threshold self.line_minlen
    # 需要可视化时包含的参数
    # mask
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def detect(self) -> tuple:
        if 3 <= len(self.stack) <= self.stack_maxsize:
            diff_img = m3func(self.stack)
        else:
            return False, [], self.stack[-1]
            #diff_img = m3func(self.stack,
            #                  getattr(self, "median_sampling_num", 1))
        diff_img = cv2.medianBlur(diff_img, 3)
        _, dst = cv2.threshold(diff_img, self.bi_threshold, 255,
                               cv2.THRESH_BINARY)
        # TODO: 这一套对噪点大的不太适用。。。
        dst = cv2.dilate(
            dst,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        )
        dst = cv2.morphologyEx(
            dst, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
        linesp = cv2.HoughLinesP(np.array(dst, dtype=np.uint8), 1, pi,
                                 self.line_threshold, self.line_minlen, 0)

        if linesp is None:
            return False, [], dst
        return True, linesp[0], dst

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


class FastDetector(BaseDetector):
    '''基于日本人版本改写的更快更简洁的检测器。
    拟用于对多帧输入（或者可视为单个输入具有长曝光的）实现检测。
    '''

    # 必须包含的参数
    # bi_threshold line_threshold self.line_minlen
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 2帧窗口（硬编码）
        self.stack_maxsize = 2

    def detect(self):
        # 短于4帧时不进行判定
        if len(self.stack) < self.stack_maxsize:
            return False, []
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
        self.linesp = cv2.HoughLinesP(dst, 1, pi, self.line_threshold,
                                      self.line_minlen, 0)
        if self.linesp is None:
            return False, []
        return True, self.linesp[0]
