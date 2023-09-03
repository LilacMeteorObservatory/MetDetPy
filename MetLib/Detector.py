import cv2
import numpy as np

from .utils import EMA, generate_group_interpolate

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

    raise NotImplementedError(f"Unimplement Detector: {name}")


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
    h, w, c = src.shape
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


class RefEMA(EMA):
    """使用滑动窗口平均机制，用于维护参考背景图像和评估信噪比的模块。

    Args:
        EMA (_type_): _description_
    """

    def __init__(self, n, ref_mask, area=0) -> None:
        super().__init__(n)
        h, w, c = ref_mask.shape
        self.ref_img = np.zeros_like(ref_mask, dtype=float)
        self.img_stack = np.zeros((n, h, w, c), dtype=np.uint8)
        self.std_interval = 2 * n
        self.timer = 0
        self.noise = 0
        if area == 0:
            self.est_std = np.std
            self.signal_ratio = ((h * w) / np.sum(ref_mask))**(1 / 2)
        else:
            self.est_std, self.signal_ratio = self.select_subarea(ref_mask,
                                                                  area=area)

    def update(self, new_frame):
        self.timer += 1
        # 更新移动窗栈与均值背景参考图像（ref_img）
        #self.img_stack.append(new_frame)
        #if len(self.img_stack) >= self.n:
        #    self.ref_img -= self.img_stack.pop(0)
        #self.ref_img += new_frame
        #logger(np.mean(self.ref_img))

        # 更新移动窗栈与均值背景参考图像（ref_img）
        rep_id = (self.timer - 1) % self.n
        if self.timer > self.n:
            self.ref_img -= self.img_stack[rep_id]
        # update new frame to the place.
        self.img_stack[rep_id] = new_frame
        self.ref_img += self.img_stack[rep_id]

        # 每std_interval时间更新一次std
        if self.timer % self.std_interval == 0:
            noise = self.est_std(self.img_stack[:self.length] - self.bg_img)
            self.noise = noise
            if len(self.ema_pool) == self.n:
                # 考虑到是平稳序列，引入一种滤波修正估算方差(截断大于三倍标准差的更新...)
                # kinda trick
                noise = self.mean + min(self.noise - self.mean, 2 * self.std)
            super().update(noise)

    def select_subarea(self, mask, area=0.1):
        """用于选择一个尽量好的子区域评估STD。

        Args:
            mask (_type_): _description_
            area (float, optional): _description_. Defaults to 0.1.

        Returns:
            _type_: _description_
        """

        h, w, c = mask.shape
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
        self.roi = (x1, y1, x1 + sub_h, y1 + sub_w)
        return lambda imgs: np.std(imgs[:, x1:x1 + sub_h, y1:y1 + sub_w]
                                   ), light_ratio

    @property
    def bg_img(self):
        return self.ref_img / self.length

    @property
    def length(self):
        return min(self.n, self.timer)

    @property
    def li_img(self):
        return np.max(self.img_stack, axis=0)


class StdMultiAreaEMA(EMA):
    """用于评估STD的EMA，选取数个区域作为参考。每次取中值作为参考标准差。
    是原型代码。
    有点慢，不同区域可能也有不同（差异比较大）
    Args:
        EMA (_type_): _description_
    """

    def __init__(self, n, ref_mask, area=0.1, k=3) -> None:
        super().__init__(n)
        self.k = k
        self.area = area
        self.areas, self.area_ratios = self.select_topk_subarea(
            ref_mask, self.area, self.k)

    def update(self, diff_stack):
        stds = np.zeros((self.k, ))
        for i, ((l, r, t, b),
                ratio) in enumerate(zip(self.areas, self.area_ratios)):
            stds[i] = np.std(diff_stack[:, l:r, t:b]) / ratio
        #logger(stds)
        return super().update(np.median(stds))

    def select_topk_subarea(self, mask, area, topk=1):
        h, w, c = mask.shape
        sub_rate = area**(1 / 2)
        slide_n = np.floor(1 / sub_rate)
        best_cor = (slide_n - 1) / 2
        sub_h, sub_w = int(h * sub_rate), int(w * sub_rate)
        dx, dy = int(h / slide_n), int(w / slide_n)
        score_mat = np.zeros((slide_n, slide_n))
        cor_mat = np.zeros((slide_n, slide_n))
        for i in range(slide_n):
            for j in range(slide_n):
                # 得分由两个部分组成，区域有效面积和靠近中心的程度。
                # 相同得分的情况下，优先选择靠近中心的区域。
                area_mask = mask[i * dx:i * dx + sub_h, j * dy:j * dy + sub_w]
                cor_mat[i, j] = -(abs(i - best_cor) + abs(j - best_cor))
                score_mat[i, j] = np.sum(area_mask)
        top_pos = np.argpartition((score_mat + cor_mat).reshape(-1),
                                  -topk)[-topk:]
        top_x, top_y = top_pos // slide_n, top_pos % slide_n
        area_list = [(x * dx, x * dx + sub_h, y * dy, y * dy + sub_w)
                     for (x, y) in zip(top_x, top_y)]
        mask_list = [
            score_mat[x, y] / (sub_h * sub_w) for (x, y) in zip(top_x, top_y)
        ]
        return area_list, mask_list


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

    def draw_light_on_bg(self, bg, light, extra_info=[]):
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
            text_h = 20
            if getattr(self, "ref_ema", None):
                if getattr(self.ref_ema, "roi", None):
                    x1, y1, x2, y2 = self.ref_ema.roi
                    cb_img = cv2.rectangle(cb_img, (y1, x1), (y2, x2),
                                           color=(128, 64, 128),
                                           thickness=2)
                cb_img = cv2.putText(
                    cb_img,
                    f"STD:{self.ref_ema.mean:.4f}; (Real noise: {self.ref_ema.noise:.4f})",
                    (10, text_h), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0),
                    1)
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


class ClassicDetector(BaseDetector):
    '''[uzanka-based detector](https://github.com/uzanka/MeteorDetector), in python.

    by: uzanka(https://github.com/uzanka)
    '''

    # 必须包含的参数
    # bi_threshold line_threshold self.max_gap
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 4帧窗口（硬编码）
        self.stack_maxsize = 4

    def detect(self):
        # 短于4帧时不进行判定
        if len(self.stack) < self.stack_maxsize:
            return [], self.stack[-1]
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

        self.linesp = [] if self.linesp is None else self.linesp[0]
        return self.linesp, self.draw_light_on_bg(self.stack[3], dst)


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
        self.ref_area = 0
        self.mask_area = np.sum(self.img_mask)
        # 加载自适应阈值的配置参数
        if self.adaptive_bi_thre:
            self.sensitivity = self.bi_cfg["sensitivity"]
            self.std2thre = sensi_func[self.sensitivity]
            self.bi_threshold = absolute_sensitivity_mapping[self.sensitivity]
            self.ref_area = self.bi_cfg["area"]
        # 使用RefEMA作为滑窗数值管理器
        self.ref_ema = RefEMA(n=self.stack_maxsize,
                              ref_mask=self.img_mask,
                              area=self.ref_area)
        # 如果启用动态蒙版（dynamic mask），在此处构建另一个滑窗管理
        if self.dynamic_mask:
            self.dy_mask_list = RefEMA(n=self.stack_maxsize,
                                       ref_mask=self.img_mask,
                                       area=self.ref_area)

    def update(self, new_frame):
        self.ref_ema.update(new_frame)
        if self.adaptive_bi_thre and (self.ref_ema.mean != np.inf):
            self.bi_threshold = self.std2thre(self.ref_ema.mean)

    def detect(self) -> tuple:

        # Preprocessing
        # Mainly calculate diff_img (which basically equals to max-mid)
        light_img = self.ref_ema.li_img
        diff_img = (light_img - self.ref_ema.bg_img).astype(dtype=np.uint8)
        diff_img = cv2.medianBlur(diff_img, 3)
        _, dst = cv2.threshold(diff_img, self.bi_threshold, 255,
                               cv2.THRESH_BINARY)

        # 以前曾用过下列两个函数改善误检情况，现因效果不明显，暂时转为弃用。
        #dst = cv2.medianBlur(dst, 3)
        #dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, self.cv_op)

        dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, self.cv_op)
        # TODO: 加入通道维之后需要修正dst
        if len(dst.shape) == 2: dst = dst[:, :, None]

        # if "dynamic_mask" is applied, stack and mask dst
        if self.dynamic_mask:
            self.dy_mask_list.update(dst)
            #print(np.sum(self.dy_mask_list.img_stack//255, axis=0),type(np.sum(self.dy_mask_list.img_stack//255, axis=0)), self.dy_mask_list.length)
            # TODO: 你要不要看看你在写什么.jpg
            dy_mask = cv2.threshold(
                np.sum(self.dy_mask_list.img_stack // 255,
                       axis=0,
                       dtype=np.uint8), self.dy_mask_list.length - 1, 1,
                cv2.THRESH_BINARY_INV)[-1]
            dy_mask = cv2.erode(dy_mask, self.cv_op)
            # TODO: 加入通道维之后需要修正
            if len(dy_mask.shape) == 2: dy_mask = dy_mask[:, :, None]
            dst = dy_mask * dst

        # 曾使用dilate放大微弱检测。
        # 有助于改善暗弱流星的检测，但可能会引起更多误报。
        #dst = cv2.dilate(dst, self.cv_op)

        # dynamic_gap机制
        # 根据产生的响应比例适量减少gap
        # 一定程度上能够改善对低信噪比场景的误检
        dst_sum = np.sum(dst / 255.) / self.mask_area * 100
        gap = max(0, 1 - dst_sum / self.max_allow_gap) * self.max_gap

        # 核心步骤：直线检测
        linesp = cv2.HoughLinesP(dst,
                                 rho=1,
                                 theta=pi,
                                 threshold=self.hough_threshold,
                                 minLineLength=self.min_len,
                                 maxLineGap=gap)
        linesp = np.array([]) if linesp is None else linesp[:, 0, :]

        # 如果产生的响应数目非常多（按照经验取值500），忽略该帧
        lines_num = len(linesp)
        if lines_num > 500:
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
            texts.append(("WARNING: TOO MANY LINES!", (0, 0, 255)))
        return linesp, self.draw_light_on_bg(light_img, dst, extra_info=texts)