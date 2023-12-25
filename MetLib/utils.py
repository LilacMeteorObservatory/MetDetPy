import datetime
import warnings
from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import cv2
import numpy as np

from .MetLog import get_default_logger

EPS = 1e-2
PI = np.pi / 180.0
VERSION = "V2.0.0_alpha"

pt_len_xy = lambda pt1, pt2: (pt1[1] - pt2[1])**2 + (pt1[0] - pt2[0])**2
drct = lambda pts: np.arccos((pts[1][1] - pts[0][1]) /
                             (pt_len_xy(pts[0], pts[1]))**(1 / 2))
drct_line = lambda pts: np.arccos((pts[3] - pts[1]) /
                                  (pt_len_xy(pts[:2], pts[2:]))**(1 / 2))
logger = get_default_logger()


def pt_offset(pt, offset) -> list:
    assert len(pt) == len(offset)
    return [value + offs for value, offs in zip(pt, offset)]


class Transform(object):
    """图像变换方法的集合类。
    """

    @classmethod
    def opencv_resize(cls, dsize, **kwargs) -> Callable:
        interpolation = kwargs.get("resize_interpolation", cv2.INTER_LINEAR)
        return partial(cv2.resize, dsize=dsize, interpolation=interpolation)

    @classmethod
    def mask_with(cls, mask) -> Callable:
        return lambda img: img * mask

    @classmethod
    def opencv_BGR2GRAY(cls) -> Callable:
        return partial(cv2.cvtColor, code=cv2.COLOR_BGR2GRAY)

    @classmethod
    def opencv_RGB2GRAY(cls) -> Callable:
        return partial(cv2.cvtColor, code=cv2.COLOR_RGB2GRAY)

    @classmethod
    def expand_3rd_channel(cls, num) -> Callable:
        """将单通道灰度图像通过Repeat方式映射到多通道图像。
        """
        assert isinstance(num, int)
        if num == 1:
            return lambda img: img[:, :, None]
        return lambda img: np.repeat(img[:, :, None], num, axis=-1)

    @classmethod
    def opencv_binary(cls, threshold, maxval=255, inv=False) -> Callable:

        def opencv_threshold_1ret(img):
            return cv2.threshold(
                img,
                thresh=threshold,
                maxval=maxval,
                type=cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY)[-1]

        return opencv_threshold_1ret

    @classmethod
    def compose(cls, trans_func: List[Callable]) -> Callable:
        """接受一个函数的列表，返回一个compose的函数，顺序执行指定的变换。

        Args:
            trans_func (List[Callable]): _description_
        """

        def transform_img(img):
            for func in trans_func:
                img = func(img)
            return img

        return transform_img


class MergeFunction(object):
    """图像变换方法的集合类。
    """

    @classmethod
    def not_merge(cls, image_stack):
        return image_stack[0]

    @classmethod
    def max(cls, image_stack):
        return np.max(image_stack, axis=0, keepdims=True)[0]

    @classmethod
    def m3func(cls, image_stack):
        """M3 for Max Minus Median.
        Args:
            image_stack (ndarray)
        """
        sort_stack = np.sort(image_stack, axis=0)
        return sort_stack[-1] - sort_stack[len(sort_stack) // 2]

    @classmethod
    def mix_max_median_stacker(cls, image_stack, threshold=80):
        img_mean = np.mean(image_stack, axis=0)
        img_max = np.max(image_stack, axis=0)
        img_max[img_max < threshold] = img_mean[img_max < threshold]
        return img_max


class SlidingWindow(object):
    """ 
    # SlidingWindow
    
    A sliding window manager, which also provides statistic for regular data struct (that are supported by numpy (ndarray, int, float, etc.)).
    """

    def __init__(self,
                 n: int,
                 size: Union[list, tuple, np.ndarray],
                 dtype: Type = int,
                 force_int: bool = False,
                 calc_std: bool = False) -> None:
        """_summary_

        Args:
            n (int): _description_
            size (Union[list, tuple, np.ndarray]): _description_
            dtype (Type, optional): _description_. Defaults to int.
            force_int (bool, optional): 启用强制整数运算加速. Defaults to False.
            calc_std (bool, optional): _description_. Defaults to False.
        """
        self.n = n
        self.timer = 0
        self.size = size
        self.cur_index = 0
        self.dtype = dtype
        self.force_int = force_int
        self.calc_std = calc_std
        sum_dtype = float
        if self.force_int and dtype == np.uint8:
            sum_dtype = np.uint32
        self.sum = np.zeros(size, dtype=sum_dtype)

        if calc_std:
            self.square_sum = np.zeros(size, dtype=sum_dtype)

        self.sliding_window = np.zeros(shape=(n, ) + tuple(size),
                                       dtype=self.dtype)

    def update(self, new_frame):
        self.timer += 1
        self.cur_index = (self.timer - 1) % self.n

        # 更新滑窗及维护求和值
        if self.timer > self.n:
            self.sum -= self.sliding_window[self.cur_index]
            if self.calc_std:
                self.square_sum -= np.square(
                    self.sliding_window[self.cur_index], dtype=np.uint32)

        self.sliding_window[self.cur_index] = new_frame
        self.sum += self.sliding_window[self.cur_index]
        if self.calc_std:
            self.square_sum += np.square(self.sliding_window[self.cur_index],
                                         dtype=np.uint32)

    @property
    def mean(self):
        if self.force_int:
            return np.array(self.sum // self.length, dtype=self.dtype)
        return self.sum / self.length

    @property
    def length(self):
        return min(self.n, self.timer)

    @property
    def max(self)->Union[int, float, np.ndarray]:
        return np.max(self.sliding_window, axis=0)

    @property
    def std(self):
        # sqrt((∑x^2 - n*avg(x)^2)/n)
        assert self.calc_std, "calc_std should be applied when initialized."
        if self.force_int:
            return np.sqrt(
                np.mean(
                    (self.square_sum - np.square(self.sum) // self.length) //
                    self.length))
        else:
            return np.sqrt(
                np.mean((self.square_sum - np.square(self.sum) / self.length) /
                        self.length))


class EMA(object):
    """
    ## 移动指数平均
    可用于对平稳序列的评估。

    Args:
        momentum (float, optional): 移动指数平均的动量. Defaults to 0.99.
        warmup (bool, optional): 是否对动量WARMUP. Defaults to True.
    """

    def __init__(self,
                 momentum: float = 0.99,
                 warmup_speed: Union[int, float] = 1) -> None:
        """
        ## 移动指数平均
        可用于对平稳序列的评估。

        Args:
            momentum (float, optional): 移动指数平均的动量. Defaults to 0.99.
            warmup (bool, optional): 是否对动量WARMUP. 设置为0时不进行warmup。
        """
        assert 0 <= momentum <= 1, "momentum should be [0,1]"
        self.init_momentum = momentum
        self.cur_momentum = momentum
        self.cur_value = 0
        self.t = 0
        self.warmup_speed = warmup_speed

    def update(self, value: float) -> None:
        if self.warmup_speed:
            self.adjust_weight()
        self.cur_value = self.cur_momentum * self.cur_value + (
            1 - self.cur_momentum) * value
        self.t += 1

    def adjust_weight(self) -> None:
        if self.t * (1 - self.init_momentum) * self.warmup_speed < 1:
            self.cur_momentum = self.init_momentum * (
                1 - (1 - self.t *
                     (1 - self.init_momentum) * self.warmup_speed)**2)
        else:
            # 结束冷启动，关闭warmup
            self.warmup_speed = 0
            self.cur_momentum = self.init_momentum


def sigma_clip(sequence, sigma=3.00):
    """Sigma裁剪均值。

    Args:
        sequence (_type_): _description_
        sigma (float, optional): _description_. Defaults to 3.00.

    Returns:
        _type_: _description_
    """
    mean, std = np.mean(sequence), np.std(sequence)
    while True:
        # update sequence
        sequence = sequence[np.abs(mean - sequence) <= sigma * std]
        updated_mean, updated_std = np.mean(sequence), np.std(sequence)
        if updated_mean == mean:
            return sequence
        mean, std = updated_mean, updated_std


def parse_resize_param(tgt_wh: Union[None, list, str, int],
                       raw_wh: Union[list, tuple]):
    # (该函数返回的wh是OpenCV风格的，即w, h)
    #TODO: fix poor English
    if tgt_wh == None:
        return list(raw_wh)
    w, h = raw_wh
    if isinstance(tgt_wh, str):
        try:
            # if str, tgt_wh is from args.
            if "x" in tgt_wh.lower():
                tgt_wh = list(map(int, tgt_wh.lower().split("x")))
            else:
                tgt_wh = int(tgt_wh)
        except Exception as e:
            raise Exception(
                f"{e}: unexpected values for argument \"--resize\".\
                 Input should be either one integer or two integers separated by \"x\"."
                % tgt_wh)
    if isinstance(tgt_wh, int):
        tgt_wh = [tgt_wh, -1] if w > h else [-1, tgt_wh]
    if isinstance(tgt_wh, (list)):
        assert len(tgt_wh) == 2, "2 values expected, got %d." % len(tgt_wh)
        # replace default value
        if tgt_wh[0] <= 0 or tgt_wh[1] <= 0:
            if tgt_wh[0] <= 0 and tgt_wh[1] <= 0:
                warnings.warn("Invalid param. Raw resolution will be used.")
                return list(raw_wh)
            idn = 0 if tgt_wh[0] <= 0 else 1
            idx = 1 - idn
            tgt_wh[idn] = int(raw_wh[idn] * tgt_wh[idx] / raw_wh[idx])
        return list(tgt_wh)
    raise TypeError(
        "Unsupported arg type: it should be <int,str,list>, got %s" %
        type(tgt_wh))


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
            f"Unsupported suffix \"{suffix}\"; Only .png and .jpeg/.jpg are supported."
        )
    status, buf = cv2.imencode(ext, img, params)
    if status:
        with open(
                filename,
                mode='wb',
        ) as f:
            f.write(buf)  # type: ignore
    else:
        raise Exception("imencode failed.")


def save_video(video_series, fps, video_path):
    cv_writer = None
    try:
        real_size = list(reversed(video_series[0].shape[:2]))
        cv_writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"MJPG"),  # type: ignore
            fps,  # type: ignore
            real_size)
        for clip in video_series:
            p = cv_writer.write(clip)
    finally:
        if cv_writer:
            cv_writer.release()


def load_8bit_image(filename):
    return cv2.imdecode(np.fromfile(filename, dtype=np.uint8),
                        cv2.IMREAD_UNCHANGED)


def transpose_wh(size_mat) -> list:
    """
    Convert OpenCV style size (width, height, (channel)) to Numpy style size (height, width, (channel)), vice versa.
    """
    if len(size_mat) == 2:
        return [size_mat[1], size_mat[0]]
    elif len(size_mat) == 3:
        x, y, c = size_mat
        return [y, x, c]
    raise Exception(
        f"size list should have length of 2 or 3, got {len(size_mat)}.")


def frame2ts(frame, fps):
    return datetime.datetime.strftime(
        datetime.datetime.utcfromtimestamp(frame / fps), "%H:%M:%S.%f")[:-3]


def ts2frame(time: str, fps: float) -> int:
    """Transfer a utc time string (format in `HH:MM:SS` or `HH:MM:SS.MS`) into the frame num.
    
    I implement this only because `datetime.timestamp() `
    seems does not support my data.
    
    (Or maybe because it is default to be started from 1971?)
    
    Notice: this function does not support time that longer than 24h.

    Args:
        time (str): UTC time string.
        fps (float): frame per second of the video.

    Returns:
        int: the corresponding frame num of the input time.
        
    Example:
        ts2frame("00:00:02.56",25) -> 64(=(2+(56/100))*25))
    """
    assert time.count(
        ":"
    ) == 2, f"Invaild time string: \":\" in \"{time}\" should appear exactly 2 times."
    if "." in time:
        dt = datetime.datetime.strptime(time, "%H:%M:%S.%f")
    else:
        dt = datetime.datetime.strptime(time, "%H:%M:%S")
    dt_time = dt.hour * 60**2 + dt.minute * 60**1 + dt.second + dt.microsecond / 1e6
    return int(dt_time * fps)


def time2frame(time: int, fps: float) -> int:
    """convert time (in ms) to the frame num.

    Args:
        time (int): time in ms.
        fps (float): video fps

    Returns:
        int: the frame num.
    """
    return int(time / 1000 * fps)


def frame2time(frame: int, fps: float) -> int:
    """convert time (in ms) to the frame num.

    Args:
        time (int): time in ms.
        fps (float): video fps

    Returns:
        int: the frame num.
    """
    return int(frame * 1000 / fps)


def timestr2int(time: str) -> int:
    """A wrapper of `ts2frame`, is mainly used to turn time-string to its corresponding integer in ms.
    
    It supports input like:
        'NoneType' -> return 'NoneType' as well.
        
        `HH:MM:SS` or `HH:MM:SS.MS` -> return its corresponding integer in ms.
        
        A string that describes the time in ms -> its corresponding integer.

    Args:
        time (str): time string.

    Returns:
        int: corresponding integer in ms.
    """
    if ":" in time:
        return ts2frame(time, fps=1000)
    return int(time)


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
        i = max(int((x - EPS) / gap), 0)
        dx = x / gap - i
        return list(map(int, inte_func[i](dx, i)))

    return color_interpolate_func


def drct_std(lines):
    """计算方向的方差。可能不完全对？

    Returns:
        _type_: _description_
    """
    drct_list = [drct_line(line) for line in lines]
    drct_copy = np.array(drct_list.copy())
    std1 = np.std(
        np.sort(drct_copy)[:-1]) if len(drct_copy) >= 3 else np.std(drct_copy)
    drct_copy[drct_copy > np.pi / 2] -= np.pi
    std2 = np.std(
        np.sort(drct_copy)[:-1]) if len(drct_copy) >= 3 else np.std(drct_copy)
    return np.min([std1, std2])


def lineset_nms(lines, max_dist, drct_prob_func):
    """对线段合集执行NMS。
    （此处的NMS并不是纯粹的去重，还包含了合并近邻直线，用于处理面积类型）
    （如何划分两种类型，目前并没有很好的想法）

    Args:
        drct: 方向方差，可作为Area贡献或者是Line的参考依据。
        lines (_type_): _description_
        drct_prob_func 计算drct属于line概率的函数
    """
    num_line = len(lines)
    length = np.sqrt(
        np.power((lines[:, 3] - lines[:, 1]), 2) +
        np.power((lines[:, 2] - lines[:, 0]), 2))
    centers = (lines[:, 2:] + lines[:, :2]) // 2
    merged_list = []
    nms_mask = np.zeros((num_line, ), dtype=np.uint8)
    length_sort = np.argsort(length)[::-1]
    lines = lines[length_sort]
    centers = centers[length_sort]

    # BFS聚类
    for i in range(num_line):
        if nms_mask[i]: continue
        nms_mask[i] = 1
        this_list = [i]
        ind = 0
        while ind < len(this_list):
            for j in range(num_line):
                if nms_mask[j]: continue
                if pt_len_xy(centers[this_list[ind]], centers[j]) < max_dist:
                    this_list.append(j)
                    nms_mask[j] = 1
            ind += 1
        merged_list.append(this_list)

    line_type = [
        "area" if drct_prob_func(drct_std(lines[x])) < 1 else "line"
        for x in merged_list
    ]

    ret_list = []
    for single_type, inds in zip(line_type, merged_list):
        if single_type == "area":
            # 找到各个边界的代表点以及整个分布的中心点作为代表点集？
            # 有点奇怪
            sin_lines = lines[inds]
            l1 = sin_lines[np.argmin(sin_lines[:, 0])]
            l2 = sin_lines[np.argmin(sin_lines[:, 1])]
            l3 = sin_lines[np.argmax(sin_lines[:, 2])]
            l4 = sin_lines[np.argmax(sin_lines[:, 3])]
            cc = np.mean((sin_lines[:, :2] + sin_lines[:, 2:]) / 2,
                         axis=0).astype(np.int16)
            #print(l1,l2,l3,l4,cc)
            ret_list.append(
                np.concatenate([l1, l2, l3, l4, cc], axis=0).reshape((-1, 2)))
        else:
            ret_list.append(lines[inds[0]])
            #drct_mean = np.mean([drct(line) for line in inds])

    return line_type, ret_list


def generate_group_interpolate(lines):
    """生成所有线段的插值点坐标，可用于计算得分。

    Args:
        lines (_type_): _description_
    """
    dxys = lines[:, 2:] - lines[:, :2]
    nums = np.max(np.abs(dxys), axis=1)
    coord_list = [[] for i in range(len(lines))]
    for i, (num, line) in enumerate(zip(nums, lines)):
        step_x, step_y = float(line[2] - line[0]) / num, float(line[3] -
                                                               line[1]) / num
        xx = np.ones(
            (num, ),
            dtype=np.int16) * line[0] if line[0] == line[2] else np.arange(
                line[0], line[2] + step_x, step=step_x).astype(np.int16)
        yy = np.ones(
            (num, ),
            dtype=np.int16) * line[1] if line[1] == line[3] else np.arange(
                line[1],
                line[3] + step_y,
                step=step_y,
            ).astype(np.int16)
        shorter = min(len(xx), len(yy))
        xx = xx[:shorter]
        yy = yy[:shorter]
        coord_list[i] = [xx, yy]
    return coord_list


def output_meteors(update_info):
    # is this necessary?
    logger = get_default_logger()
    met_lst, drop_lst = update_info
    for met in met_lst:
        logger.meteor(met)
    for met in drop_lst:
        logger.dropped(met)