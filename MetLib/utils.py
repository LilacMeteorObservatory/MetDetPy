import datetime
import sys
import warnings
from functools import partial
from math import floor

import cv2
import numpy as np

from .VideoWarpper import OpenCVVideoWarpper

eps = 1e-2

img_max = partial(np.max, axis=0)


class Munch(object):

    def __init__(self, idict) -> None:
        for (key, value) in idict.items():
            #if isinstance(value,dict):
            #    value = Munch(value)
            setattr(self, key, value)


class EMA(object):

    def __init__(self, n) -> None:
        self.n = n
        self.ema_pool = []

    def update(self, num):
        self.ema_pool.append(num)
        self.ema_pool = self.ema_pool[-self.n:]

    @property
    def mean(self):
        return np.mean(self.ema_pool) if len(self.ema_pool) > 0 else np.inf

    @property
    def std(self):
        return np.std(self.ema_pool) if len(self.ema_pool) > 0 else np.inf


class RefEMA(EMA):
    """使用滑动窗口平均机制，用于维护参考背景图像和评估信噪比的模块。

    Args:
        EMA (_type_): _description_
    """

    def __init__(self, n, ref_mask, area=0) -> None:
        super().__init__(n)
        h, w = ref_mask.shape
        self.ref_img = np.zeros_like(ref_mask, dtype=float)
        self.img_stack = np.zeros((n, h, w), dtype=np.uint8)
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
        #print(np.mean(self.ref_img))

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

        h, w = mask.shape
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
        #print(stds)
        return super().update(np.median(stds))

    def select_topk_subarea(self, mask, area, topk=1):
        h, w = mask.shape
        sub_rate = area**(1 / 2)
        slide_n = floor(1 / sub_rate)
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


def sigma_clip(sequence, sigma=3.00):
    mean, std = np.mean(sequence), np.std(sequence)
    while True:
        # update sequence
        sequence = sequence[np.abs(mean - sequence) <= sigma * std]
        updated_mean, updated_std = np.mean(sequence), np.std(sequence)
        if updated_mean == mean:
            return sequence
        mean, std = updated_mean, updated_std


def parse_resize_param(tgt_wh, raw_wh):
    #TODO: fix poor English
    w, h = raw_wh
    if isinstance(tgt_wh, str):
        try:
            # if str, tgt_wh is from args.
            if ":" in tgt_wh:
                tgt_wh = list(map(int, tgt_wh.split(":")))
            else:
                tgt_wh = int(tgt_wh)
        except Exception as e:
            raise e("Unexpected values for argument \"--resize\".\
                 Input should be either one integer or two integers separated by \":\"."
                    % tgt_wh)
    if isinstance(tgt_wh, int):
        tgt_wh = [tgt_wh, -1] if w > h else [-1, tgt_wh]
    if isinstance(tgt_wh, (list)):
        assert len(tgt_wh) == 2, "2 values expected, got %d." % len(tgt_wh)
        # replace default value
        if tgt_wh[0] <= 0 or tgt_wh[1] <= 0:
            if tgt_wh[0] <= 0 and tgt_wh[1] <= 0:
                warnings.warn("Invalid param. Raw resolution will be used.")
                return raw_wh
            idn = 0 if tgt_wh[0] <= 0 else 1
            idx = 1 - idn
            tgt_wh[idn] = int(raw_wh[idn] * tgt_wh[idx] / raw_wh[idx])
        return tgt_wh
    raise TypeError(
        "Unsupported arg type: it should be <int,str,list>, got %s" %
        type(tgt_wh))


def load_video_and_mask(video_name, mask_name=None, resize_param=(0, 0)):
    # OpenCVVideoWarpper is the default video capture source.
    # If you want to use your own way to load video, define your VideoWarpper
    # in MetLib/VideoWarpper.py and replace all OpenCVVideoWarpper in your code.
    video = OpenCVVideoWarpper(video_name)
    # Calculate real resize parameters.
    resize_param = parse_resize_param(resize_param, video.size)
    mask = None
    if mask_name:
        mask = load_mask(mask_name, resize_param)
    else:
        # If no mask_name is provided, use a all-pass mask.
        mask = np.ones((resize_param[1], resize_param[0]), dtype=np.uint8)
    return video, mask


def save_img(img, filename, quality, compressing):
    if filename.upper().endswith("PNG"):
        ext = ".png"
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), compressing]
    elif filename.upper().endswith("JPG") or filename.upper().endswith("JPEG"):
        ext = ".jpg"
        params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    else:
        suffix = filename.split(".")[-1]
        raise NameError(
            f"Unsupported suffix \"{suffix}\";\Only .png and .jpeg/.jpg are supported."
        )
    status, buf = cv2.imencode(ext, img, params)
    if status:
        with open(
                filename,
                mode='wb',
        ) as f:
            f.write(buf)
    else:
        raise Exception("imencode failed.")


def save_video(video_series, fps, video_path):
    try:
        real_size = list(reversed(video_series[0].shape[:2]))
        cv_writer = cv2.VideoWriter(video_path,
                                    cv2.VideoWriter_fourcc(*"MJPG"), fps,
                                    real_size)
        for clip in video_series:
            p = cv_writer.write(clip)
    finally:
        cv_writer.release()


def load_mask(filename, resize_param):
    mask = cv2.imdecode(np.fromfile(filename, dtype=np.uint8),
                        cv2.IMREAD_UNCHANGED)
    if filename.lower().endswith(".jpg"):
        mask = preprocessing(mask, resize_param=resize_param)
        return cv2.threshold(mask, 128, 1, cv2.THRESH_BINARY)[-1]
    elif filename.lower().endswith(".png"):
        mask = cv2.resize(mask[:, :, -1], resize_param)
        return cv2.threshold(mask, 128, 1, cv2.THRESH_BINARY_INV)[-1]


def preprocessing(frame, mask=1, resize_param=(0, 0)):
    frame = cv2.cvtColor(cv2.resize(frame, resize_param), cv2.COLOR_BGR2GRAY)
    return frame * mask


def m3func(image_stack):
    """M3 for Max Minus Median.
    Args:
        image_stack (ndarray)
    """
    sort_stack = np.sort(image_stack, axis=0)
    return sort_stack[-1] - sort_stack[len(sort_stack) // 2]


def mix_max_median_stacker(image_stack, threshold=80):
    img_mean = np.mean(image_stack, axis=0)
    img_max = np.max(image_stack, axis=0)
    img_max[img_max < threshold] = img_mean[img_max < threshold]
    return img_max


def _rf_est_kernel(video_loader, n_frames):
    try:
        video_loader.start()
        f_sum = np.zeros((n_frames, ), dtype=float)
        for i in range(n_frames):
            if not video_loader.stopped:
                frame = video_loader.pop()
                f_sum[i] = np.sum(frame)
            else:
                f_sum = f_sum[:i]
                break
        # mean filter with windows_size=3
        #f_sum = np.array([(
        #    f_sum[max(0, i - 1)] + f_sum[i] + f_sum[min(len(f_sum) - 1, i + 1)]
        #) / 3 for i, _ in enumerate(f_sum)])

        A0, A1, A2, A3 = f_sum[:-3], f_sum[1:-2], f_sum[2:-1], f_sum[3:]

        diff_series = f_sum[1:] - f_sum[:-1]
        rmax_pos = np.where((2 * A2 - (A1 + A3) > 0) & (2 * A1 - (A0 + A2) < 0)
                            & (np.abs(diff_series[1:-1]) > 0.01))[0]
        #plt.scatter(rmax_pos + 1, diff_series[rmax_pos + 1], s=30, c='r')
        #plt.plot(diff_series, 'r')
        #plt.show()
    finally:
        video_loader.stop()
    return rmax_pos[1:] - rmax_pos[:-1]


def rf_estimator(video_loader):
    """用于为给定的视频估算实际的曝光时间。

    部分相机在录制给定帧率的视频时，可以选择慢于帧率的单帧曝光时间（慢门）。
    还原真实的单帧曝光时间可帮助更好的检测。
    但目前没有做到很好的估计。

    Args:
        video (cv2.VideoCapture): 给定视频片段。
        mask (ndarray): the mask for the video.
    """
    total_frame = video_loader.iterations
    start_frame = video_loader.start_frame
    if total_frame < 300:
        # 若不超过300帧 则进行全局估算
        intervals = _rf_est_kernel(video_loader, total_frame)
    else:
        # 超过300帧的 从开头 中间 结尾各抽取100帧长度的视频进行估算。
        video_loader.reset(iterations=100)
        intervals_1 = _rf_est_kernel(video_loader, 100)

        video_loader.reset(start_frame=start_frame + (total_frame - 100) // 2,
                           iterations=100)
        intervals_2 = _rf_est_kernel(video_loader, 100)

        video_loader.reset(start_frame=start_frame + total_frame - 100,
                           iterations=100)
        intervals_3 = _rf_est_kernel(video_loader, 100)

        intervals = np.concatenate([intervals_1, intervals_2, intervals_3])
    if len(intervals) == 0:
        return 1
    # 非常经验的取值方法...
    est_frames = np.round(
        min(np.median(intervals), np.mean(sigma_clip(intervals))))
    return est_frames


def init_exp_time(exp_time, video_loader, upper_bound=0.25):
    """Init exposure time. Return the exposure time that gonna be used in MergeStacker.
    (SimpleStacker do not rely on this.)

    Args:
        exp_time (int,float,str): value from config.json. It can be either a value or a specified string.
        video (cv2.VideoCapture): the video.
        mask (np.array): mask array.

    Raises:
        ValueError: raised if the exp_time is invalid.

    Returns:
        exp_time: the exposure time in float.
    """
    # TODO: Rewrite this annotation.
    fps = video_loader.video.fps
    assert isinstance(
        exp_time, (str, float, int)
    ), "exp_time should be either <str, float, int>, got %s" % (type(exp_time))
    if isinstance(exp_time, str):
        if exp_time == "real-time":
            return 1 / fps
        if exp_time == "slow":
            # TODO: Any better idea?
            return 1 / 4
        if exp_time == "auto":
            rf = rf_estimator(video_loader)
            if rf / fps > upper_bound:
                print(
                    Warning(
                        f"Warning: Unexpected exposuring time (too long):{rf/fps:.2f}s. Use {upper_bound:.2f}s instead."
                    ))
            return min(rf / fps, upper_bound)
        try:
            exp_time = float(exp_time)
        except ValueError as E:
            raise ValueError(
                "Invalid exp_time string value: It should be selected from [float], [int], "
                + "real-time\",\"auto\" and \"slow\", got %s." % (exp_time))
    if isinstance(exp_time, (float, int)):
        if exp_time * fps < 1:
            print(
                Warning(
                    f"Warning: Invalid exposuring time (too short). Use {1/fps:.2f}s instead."
                ))
            return 1 / fps
        return float(exp_time)


def frame2ts(frame, fps):
    return datetime.datetime.strftime(
        datetime.datetime.utcfromtimestamp(frame / fps), "%H:%M:%S.%f")[:-3]


def color_interpolater(color_list):
    # 用于创建跨越多种颜色的插值条
    # 返回一个函数，该函数可以接受[0,1]并返回对应颜色
    nums = len(color_list)
    color_list = list(map(np.array, color_list))
    gap = 1 / (nums - 1)
    inte_func = []
    for i in range(nums - 1):
        inte_func.append(lambda x, i: np.array(
            (1 - x) * color_list[i] + x * color_list[i + 1], dtype=np.uint8))

    def color_interpolate_func(x):
        i = max(int((x - eps) / gap), 0)
        dx = x / gap - i
        return list(map(int, inte_func[i](dx, i)))

    return color_interpolate_func


def least_square_fit(pts):
    """fit pts to the linear func Ax+By+C=0

    Args:
        pts (_type_): _description_

    Returns:
        tuple: parameters (A,B,C)
        float: average loss
    """
    pts = np.array(pts)
    avg_xy = np.mean(pts, axis=0)
    dpts = pts - avg_xy
    dxx, dyy = np.sum(dpts**2, axis=0)
    dxy = np.sum(dpts[:, 0] * dpts[:, 1])
    v, m = np.linalg.eig([[dxx, dxy], [dxy, dyy]])
    A, B = m[np.argmin(v)]
    # fixed
    # TODO: this works fine now. but why??
    B = -B

    C = -np.sum(np.array([A, B]) * avg_xy)
    loss = np.mean(np.abs(np.einsum("a,ba->b", np.array([A, B]), pts) + C))
    return (A, B, C), loss


#####################################################
##WARNING: The Following Functions Are Deprecatred.##
#####################################################


def GammaCorrection(src, gamma):
    """deprecated."""
    return np.array((src / 255.)**(1 / gamma) * 255, dtype=np.uint8)


#@numba.jit(nopython=True, fastmath=True, parallel=True)
def sanaas(stack: np.array, des: np.array, boolmap, L, H, W):
    '''
    ================================DEPRECATED========================
    stack [list[np.array]],
    new_matrix H*W
    des[list[np.array]] L*H*W stack[des]是真实升序序列。des是下标序列。
    已实现可借助numba的加速版本；但令人悲伤的仍然是其运行速度。因此废弃。
    ================================DEPRECATED========================
    '''
    # 双向冒泡
    # stack and des : L*(HW)
    for k in np.arange(0, L - 1, 1):
        # boolmap : (HW,) (bool)
        for i in np.arange(H * W):
            boolmap[i] = stack[des[k, i], i] > stack[des[k + 1, i], i]
        des[k, np.where(boolmap)[0]], des[k + 1, np.where(boolmap)[0]] = des[
            k + 1, np.where(boolmap)[0]], des[k, np.where(boolmap)[0]]

    for k in np.arange(L - 2, -1, -1):
        # boolmap : (HW,) (bool)
        for i in np.arange(H * W):
            boolmap[i] = stack[des[k, i], i] > stack[des[k + 1, i], i]
        for position in np.where(boolmap)[0]:
            des[k, position], des[k + 1,
                                  position] = des[k + 1,
                                                  position], des[k, position]
    return stack, des


def series_keeping(sort_stack, frame, window_size, boolmap, L, H, W):
    sort_stack = np.concatenate((sort_stack[1:], sort_stack[:1]), axis=0)
    # Reshape为二维
    window_stack = np.reshape(window_stack, (L, H * W))
    sort_stack = np.reshape(sort_stack, (L, H * W))
    # numba加速的双向冒泡
    window_stack, sort_stack = sanaas(window_stack, sort_stack, boolmap, L, H,
                                      W)
    # 计算max-median
    diff_img = np.reshape(
        window_stack[sort_stack[-1],
                     np.where(sort_stack[-1] >= 0)[0]], (H, W))
    # 形状还原
    window_stack = np.reshape(window_stack, (L, H, W))
    sort_stack = np.reshape(sort_stack, (L, H, W))

    #update and calculate
    window_stack.append(frame)
    if len(window_stack) > window_size:
        window_stack.pop(0)
    diff_img = np.max(window_stack, axis=0) - np.median(window_stack, axis=0)
    return diff_img
