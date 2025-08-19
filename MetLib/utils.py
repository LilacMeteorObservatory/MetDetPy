from __future__ import annotations

import datetime
import os.path as path
import warnings
from typing import Any, Callable, Optional, Sequence, Type, TypeVar, Union

import cv2
import numpy as np
from cv2.typing import MatLike
from easydict import EasyDict
from numpy.typing import DTypeLike, NDArray

from .metlog import get_default_logger
from .metstruct import Box

PROJECT_NAME = "MetDetPy"
VERSION = "V2.3.0"
EPS = 1e-2
PI = np.pi / 180.0
LIVE_MODE_SPEED_CTRL_CONST = 0.9
EULER_CONSTANT = 0.5772
MAX_LOOP_CNT = 10
# WORK_PATH 指向 MetDetPy 项目根目录位置。
WORK_PATH = path.split(path.dirname(path.abspath(__file__)))[0]

logger = get_default_logger()

#### Typing alias
U8Mat = Union[NDArray[np.uint8], MatLike]
FloatMat = NDArray[np.float64]
NpCollect = TypeVar("NpCollect", np.int_, np.float64)
Addable = TypeVar("Addable", int, float)

STR2DTYPE: dict[str, DTypeLike] = {
    "float32": np.float32,
    "float16": np.float16,
    "int8": np.int8
}
SWITCH2BOOL = {"on": True, "off": False}

DTYPE_UPSCALE_MAP: dict[DTypeLike, DTypeLike] = {
    np.dtype('uint8'): np.dtype('uint16'),
    np.dtype('uint16'): np.dtype('uint32'),
    np.dtype('uint32'): np.dtype('uint64'),
    np.dtype('uint64'): float
}

COLOR_MAP = {
    "black": (0, 0, 0),
    "green": (0, 255, 0),
    "orange": (0, 128, 255),
    "purple": (128, 64, 128),
    "red": (0, 0, 255),
    "white": (255, 255, 255),
    "yellow": (0, 255, 255)
}


def pt_len_sqr(pt1: Union[list[float], FloatMat], pt2: Union[list[float],
                                                             FloatMat]):
    """Return the square of the distance between two points. 
    When passing a matrix, make sure the last dim has length of 2 (like [n,2]).
    返回两点之间的距离的平方。接受ndarray时，需要最后一维形状为2。

    Args:
        pt1 (Union[np.ndarray, list, tuple]): point 1
        pt2 (Union[np.ndarray, list, tuple]): point 2

    Returns:
        Union[int, np.ndarray]: 距离的平方。
    """
    if isinstance(pt1, np.ndarray) and isinstance(pt2, np.ndarray):
        return (pt1[..., 1] - pt2[..., 1])**2 + (pt1[..., 0] - pt2[..., 0])**2
    else:
        return (pt1[1] - pt2[1])**2 + (pt1[0] - pt2[0])**2


def pt_len(pt1: Union[list[float], FloatMat], pt2: Union[list[float],
                                                         FloatMat]):
    """Return the distance between two points. 
    When passing a matrix, make sure the last dim has length of 2 (like [n,2]).
    返回两点之间的实际距离。接受ndarray时，需要最后一维形状为2。

    Args:
        pt1 (Union[np.ndarray, list, tuple]): _description_
        pt2 (Union[np.ndarray, list, tuple]): _description_

    Returns:
        Union[int, np.ndarray]: 距离。
    """
    return np.sqrt(pt_len_sqr(pt1, pt2))


def pt_drct(pt1: list[float], pt2: list[float]):
    """Return the direction of the line of two points, in [0, pi]。
    返回两点之间连线的角度值，范围为[0, pi]。

    Args:
        pt1 (Union[np.ndarray, list, tuple]): point 1
        pt2 (Union[np.ndarray, list, tuple]): point 2

    Returns:
        float: the angle(direction) of two points, in [0, pi].
    """
    return np.arccos((pt2[1] - pt1[1]) / (pt_len(pt1, pt2)))


def pt_offset(pt: Sequence[Addable], offset: Sequence[Addable]):
    assert len(pt) == len(offset)
    return [value + offs for value, offs in zip(pt, offset)]


def keep1ret_value(func: Callable[..., tuple[Any, Any]], select_pos: int):
    """A simple decorator that only keep the value of specific position.
    """

    def selector_core(args: Any, **kwargs: Any):
        res = func(args, **kwargs)
        assert len(res) > select_pos >= -len(
            res), f"selected ret at pos {select_pos},"
        " got only {len(res)} ret value."
        return res[select_pos]

    return selector_core


class Transform(object):
    """图像变换方法的集合类，及一个用于执行集成变换的方法。
    """
    MASK_FLAG = "MASK"
    PATTERN_MAPPING = {
        "BGGR": cv2.COLOR_BAYER_BGGR2BGR,
        "RGGB": cv2.COLOR_BAYER_RGGB2BGR
    }

    def __init__(self) -> None:
        self.transform: list[tuple[Callable[..., MatLike], dict[str,
                                                                Any]]] = []

    def opencv_resize(self, dsize: list[int], **kwargs: Any):
        interpolation = kwargs.get("resize_interpolation", cv2.INTER_LINEAR)
        self.transform.append(
            (cv2.resize, dict(dsize=dsize, interpolation=interpolation)))

    def opencv_BGR2GRAY(self):
        self.transform.append((cv2.cvtColor, dict(code=cv2.COLOR_BGR2GRAY)))

    def opencv_RGB2GRAY(self):
        self.transform.append((cv2.cvtColor, dict(code=cv2.COLOR_RGB2GRAY)))

    def opencv_GRAY2BGR(self):
        self.transform.append((cv2.cvtColor, dict(code=cv2.COLOR_GRAY2BGR)))

    def mask_with(self, mask: MatLike):

        def _mask_with(img: MatLike, mask: MatLike):
            return img * mask

        self.transform.append((_mask_with, dict(mask=mask)))

    def expand_3rd_channel(self, num: int):
        """将单通道灰度图像通过Repeat方式映射到多通道图像。
        """
        assert isinstance(
            num, int
        ) and num > 0, f"num invalid! expect int>0, got {num} with dtype={type(num)}."
        self.transform.append((np.expand_dims, dict(axis=-1)))
        if num > 1:
            self.transform.append((np.repeat, dict(repeats=num, axis=-1)))

    def opencv_binary(self,
                      threshold: Union[float, int],
                      maxval: int = 255,
                      inv: bool = False):
        self.transform.append(
            (keep1ret_value(cv2.threshold, select_pos=-1),
             dict(thresh=threshold,
                  maxval=maxval,
                  type=cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY)))

    def opencv_debayer(self, pattern: str = "BGGR"):
        assert pattern in self.PATTERN_MAPPING, f"unsupport debayer pattern! choice from {self.PATTERN_MAPPING}"
        self.transform.append((cv2.cvtColor, dict(code=cv2.COLOR_BGR2GRAY)))
        self.transform.append(
            (cv2.cvtColor, dict(code=self.PATTERN_MAPPING[pattern], dstCn=3)))

    def exec_transform(self, img: MatLike) -> MatLike:
        """按顺序对给定的输入执行给定的图像变换。

        Args:
            img (MatLike): 输入图像

        Returns:
            MatLike: 变换后图像
        """
        for [transform, kwargs] in self.transform:
            img = transform(img, **kwargs)
        return img


class MergeFunction(object):
    """多张图像合并方法的集合类。
    """

    @classmethod
    def not_merge(cls, image_stack: Sequence[Union[U8Mat, MatLike]]):
        return image_stack[0]

    @classmethod
    def max(cls, image_stack: Sequence[Union[U8Mat, MatLike]]):
        return np.max(image_stack, axis=0, keepdims=True)[0]

    @classmethod
    def m3func(cls, image_stack: Sequence[Union[U8Mat, MatLike]]):
        """M3 for Max Minus Median.
        Args:
            image_stack (ndarray)
        """
        sort_stack = np.sort(image_stack, axis=0)
        return sort_stack[-1] - sort_stack[len(sort_stack) // 2]

    @classmethod
    def mix_max_median_stacker(cls,
                               image_stack: Sequence[Union[U8Mat, MatLike]],
                               threshold: int = 80):
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
                 size: Sequence[int],
                 dtype: type = int,
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
        self.sum: Union[NDArray[np.float64],NDArray[np.uint32]] = np.zeros(size, dtype=sum_dtype)

        if calc_std:
            self.square_sum = np.zeros(size, dtype=sum_dtype)

        self.sliding_window: NDArray[np.uint16] = np.zeros(shape=(n, ) + tuple(size),
                                       dtype=self.dtype)

    def update(self, new_frame: U8Mat):
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
    def mean(self)->Union[NDArray[np.uint32],NDArray[np.float64]]:
        if self.force_int:
            return np.array(self.sum // self.length, dtype=self.dtype)
        return self.sum / self.length

    @property
    def length(self):
        return min(self.n, self.timer)

    @property
    def max(self) -> U8Mat:
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

    def update(self, value: Union[U8Mat, int, float]) -> None:
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


class Uint8EMA(EMA):
    """
    ## 适用于Uint8矩阵的整数移动指数平均
    用于计算图像的平稳评估。

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

    def update(self, value: U8Mat) -> None:
        if self.warmup_speed:
            self.adjust_weight()
        value_copy = np.array(value, dtype=np.int16)
        self.cur_value = (self.cur_momentum * self.cur_value +
                          (1 - self.cur_momentum) * value_copy).astype(
                              np.uint8)
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


class FastGaussianParam(object):
    """
    GaussianParam, but faster. 
    通过INT量化+优化数据储存提速，仅在输出时换算为浮点数。
    Streaming mean and variance.
    Args:
        object (_type_): _description_
    TODO: 优化接口，和普通版本统一；进一步支持float类型
    （理论可以通过后置除法+提高数据范围提高精度）
    NOTE: FastGaussianParam 能够最大累加而不溢出的数目与数量相关，因此需要在输入前手动扩大数据范围。
    TODO: 从输入端控制数据类型，避免输入端手动转换（反直觉）    
    """

    def __init__(self,
                 sum_mu: U8Mat,
                 square_num: Optional[U8Mat] = None,
                 n: Optional[U8Mat] = None,
                 ddof: int = 1,
                 dtype_n: DTypeLike = np.dtype("int16")):
        # 默认对 sum_mu
        self.sum_mu = sum_mu
        if square_num is not None:
            self.square_sum = square_num
        else:
            # var默认根据sum_mu构造而成
            sq_dtype = self.get_upscale_dtype_as(self.sum_mu)
            self.square_sum = np.square(sum_mu, dtype=sq_dtype)
        self.n = n if n is not None else np.ones_like(self.sum_mu,
                                                      dtype=dtype_n)
        self.ddof = ddof

    @property
    def mu(self) -> NDArray[np.float64]:
        return np.round(self.sum_mu / self.n)

    @property
    def var(self) -> NDArray[np.float64]:
        #D(X) = ∑((X-E(X))^2)/(n-ddof)
        #     = (∑X^2 - nE(X)^2) /(n-ddof)
        #     = (∑X^2 - (∑X)^2/n) /(n-ddof)
        sum_mu = np.array(self.sum_mu, dtype=self.square_sum.dtype)
        return (self.square_sum - np.square(sum_mu) / self.n) / (self.n -
                                                                 self.ddof)

    def upscale(self):
        upscaled_sum_mu_dtype = self.get_upscale_dtype_as(self.sum_mu)
        upscaled_sum_sq_dtype = self.get_upscale_dtype_as(self.square_sum)
        self.sum_mu = np.array(self.sum_mu, dtype=upscaled_sum_mu_dtype)
        self.square_sum = np.array(self.square_sum,
                                   dtype=upscaled_sum_sq_dtype)

    def get_upscale_dtype_as(self, ref_array: U8Mat):
        """必要时候提升数据范围
        """
        return DTYPE_UPSCALE_MAP[
            ref_array.dtype] if ref_array.dtype in DTYPE_UPSCALE_MAP else float

    def apply_zero_var(self, full_img: FastGaussianParam):
        """修复n为0的情况。应用修复。
        
        TODO: 需要长期观测该逻辑。
        """
        zero_pos = (self.n == 0)
        self.n[zero_pos] = full_img.n[zero_pos]
        self.sum_mu[zero_pos] = full_img.sum_mu[zero_pos]
        self.square_sum[zero_pos] = full_img.square_sum[zero_pos]

    def __add__(self, g2: FastGaussianParam):
        g1 = self
        assert isinstance(g2, FastGaussianParam), "unacceptable object"
        assert g1.ddof == g2.ddof, "unmatched var calculation!"
        assert g1.square_sum is not None and g2.square_sum is not None, "Invalid square num!"
        return FastGaussianParam(sum_mu=g1.sum_mu + g2.sum_mu,
                                 square_num=g1.square_sum + g2.square_sum,
                                 n=g1.n + g2.n,
                                 ddof=g1.ddof)

    def __sub__(self, g2: FastGaussianParam):
        g1 = self
        assert isinstance(g2, FastGaussianParam), "unacceptable object"
        assert g1.ddof == g2.ddof, "unmatched var calculation!"
        assert (g1.n - g2.n).any() >= 0, "generate n<0 fistribution!"
        return FastGaussianParam(sum_mu=g1.sum_mu - g2.sum_mu,
                                 square_num=g1.square_sum - g2.square_sum,
                                 n=g1.n - g2.n,
                                 ddof=g1.ddof)

    def mask(self, mask_pos: NDArray[np.bool_]):
        assert mask_pos.dtype == np.dtype("bool"), "Invalid mask!"
        self.sum_mu *= mask_pos
        self.square_sum *= mask_pos
        self.n = np.array(mask_pos, dtype=np.uint16)

    @property
    def shape(self):
        return self.sum_mu.shape


def sigma_clip(sequence: Union[list[int], NDArray[np.int_]],
               sigma: Union[float, int] = 3.00) -> NDArray[np.int_]:
    """Sigma-clipping average, return the sequence where all values are within the given sigma value.

    Args:
        sequence (Union[list, np.ndarray]): the input sequence.
        sigma (float, optional): _description_. Defaults to 3.00.

    Returns:
        np.ndarray: the output sequence.
    """
    # sequence should be flatten before execution
    sequence = np.array(sequence).reshape((-1, ))
    mean, std = np.mean(sequence), np.std(sequence)
    loop_cnt = 0
    while True:
        # update sequence
        sequence = sequence[np.abs(mean - sequence) <= sigma * std]
        updated_mean, updated_std = np.mean(sequence), np.std(sequence)
        if updated_mean == mean:
            return sequence
        mean, std = updated_mean, updated_std
        loop_cnt += 1
        if loop_cnt >= MAX_LOOP_CNT:
            return sequence


def circular_kernel(size: int) -> U8Mat:
    """
    生成一个给定大小的圆形卷积核（binary mask）。

    参数：
        size (int): 卷积核的宽高（必须是正奇数）

    返回：
        numpy.ndarray: 二维圆形核，中心为1，外围为0
    """
    if size % 2 == 0 or size <= 0:
        raise ValueError("size 必须为正奇数")

    radius = size // 2
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    mask = x**2 + y**2 <= radius**2
    return mask.astype(np.uint8)


def parse_resize_param(tgt_wh: Union[None, list[int], str, int],
                       raw_wh: Union[list[int], tuple[int, int]]) -> list[int]:
    """Parse resize tgt_wh according to the video size, and return a list includes target width and height.

    This function accepts and returns in [w,h] order (i.e. OpenCV style).

    Args:
        tgt_wh (Union[None, list, str, int]): the desired resolution. Accept examples: None; [960, 540]; "960x540"; 960
        raw_wh (Union[list, tuple]): the video size. 

    Raises:
        Exception: if the tgt_wh is invalid, an expection will be raised.
        TypeError: if the tgt_wh is not of any type that hints, an TypeError will be raised.

    Returns:
        list: the desired resolution in list of int.
    """
    # If tgt_wh is not specified, it returns the video size (no resize).
    if tgt_wh == None:
        return list(raw_wh)

    w, h = raw_wh
    # convert string to list/int
    if isinstance(tgt_wh, str):
        try:
            # if str, tgt_wh is from args.
            if "x" in tgt_wh.lower():
                tgt_wh = list(map(int, tgt_wh.lower().split("x")))
            else:
                tgt_wh = int(tgt_wh)
        except Exception as e:
            raise Exception(
                f"{e}: unexpected values for argument \"--resize\":"
                f" input should be either one integer or two integers separated by \"x\", got {tgt_wh}."
            )
    # if int, convert to list by set the short side to -1 (adaptive)
    if isinstance(tgt_wh, int):
        tgt_wh = [tgt_wh, -1] if w > h else [-1, tgt_wh]

    # parse list
    if isinstance(tgt_wh, list):
        if len(tgt_wh) != 2:
            raise Exception(
                f"Expected tgt_wh is converted to a list with 2 elements, got {len(tgt_wh)}."
            )
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
        f"Unsupported arg type: it should be <int,str,list>, got {type(tgt_wh)}."
    )


def transpose_wh(size_mat: Union[list[int], tuple[int, ...],
                                 NDArray[np.int_]]):
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


def frame2ts(frame: int, fps: float) -> str:
    return datetime.datetime.strftime(
        datetime.datetime.fromtimestamp(frame / fps, tz=datetime.timezone.utc),
        "%H:%M:%S.%f")[:-3]


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
    return int(round(dt_time * fps))


def time2frame(time: int, fps: float) -> int:
    """convert time (in ms) to the frame num.

    Args:
        time (int): time in ms.
        fps (float): video fps

    Returns:
        int: the frame num.
    """
    return int(round(time / 1000 * fps))


def frame2time(frame: int, fps: float) -> int:
    """convert the frame to time (in ms).

    Args:
        frame (int): time in ms.
        fps (float): video fps

    Returns:
        int: the time num.
    """
    return int(round(frame * 1000 / fps))


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


def color_interpolater(input_color_list: list[tuple[int,...]]):
    """
    用于创建跨越多种颜色的插值条
    返回一个函数，该函数可以接受[0,1]并返回对应颜色。
    

    Args:
        input_color_list (list[list[int]]): list of color list.
    """

    def gen_func(index: int):

        def calculate_with_color_list(dx: float):
            return np.array(
                (1 - dx) * color_list[index] + dx * color_list[index + 1],
                dtype=np.uint8)

        return calculate_with_color_list

    nums = len(input_color_list)
    color_list = list(map(np.array, input_color_list))
    gap = 1 / (nums - 1)
    inte_func: list[Callable[[float], U8Mat]] = []
    for i in range(nums - 1):
        inte_func.append(gen_func(index=i))

    def color_interpolate_func(x: float):
        if x > 1: x = 1
        if x < 0: x = 0
        i = max(int((x - EPS) / gap), 0)
        dx = x / gap - i
        return tuple(map(int, inte_func[i](dx)))

    return color_interpolate_func


def lineset_nms(
        lines: NDArray[np.int_]) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """
    Conduct NMS for line set.
    对线段合集执行NMS，并从线段集合中区分出“面积”类型。
    Args:
        lines (np.ndarray): 线段集合

    Returns:
        np.ndarray: 去重后的线段集合
    """
    # 其实也不需要阈值...每个输出都可以是百分之多少概率的直线。这会让整个概率体系更加可靠。
    # 合并线段的方法：线段的合并区域可以由其半径决定。
    # 合并概率的计算：根据合并后的长宽比
    num_line = len(lines)
    length_sqr = np.power((lines[:, 3] - lines[:, 1]), 2) + np.power(
        (lines[:, 2] - lines[:, 0]), 2)
    length_params = np.array([
        lines[:, 3] - lines[:, 1], lines[:, 0] - lines[:, 2],
        lines[:, 2] * lines[:, 1] - lines[:, 3] * lines[:, 0]
    ]).transpose()
    centers = (lines[:, 2:] + lines[:, :2]) // 2
    nms_ids = []
    nms_mask = np.zeros((num_line, ), dtype=np.uint8)
    length_sort = np.argsort(length_sqr)[::-1]
    # width_list 用于记录每个集合的宽度，在输出时给出直线比率。
    # TODO: 该机制如何与现有的流星机制结合也是一个问题。
    width_list = []
    # NMS
    for i, idx in enumerate(length_sort):
        # 如果已经被其他收纳 则忽略
        if nms_mask[idx]: continue
        # 开始新的一组
        nms_ids.append(idx)
        nms_mask[idx] = 1
        max_width = 0
        for idy in length_sort[i:]:
            if nms_mask[idy]: continue
            # 距离小于长线的length_sqr//4 (长线的半径以内) 即收纳.
            # TODO: 这个逻辑和过去并不一样。需要测试以验证稳定性。
            if pt_len_sqr(centers[idx], centers[idy]) < length_sqr[idx] // 4:
                nms_mask[idy] = 1
                # max_width only include Ax+By.
                max_width = max(
                    max_width,
                    np.abs(
                        np.sum(length_params[idx, :2] * centers[idy]) +
                        length_params[idx, -1]))
        width_list.append(max_width)

    # 后处理
    nms_lines = lines[nms_ids]
    # nonline_prob = |(Ax+By)+C|/sqrt(A^2+B^2) / LENGTH
    # *2 to calculate radius.
    nonline_prob = np.abs(width_list) / np.sqrt(
        np.sum(np.power(length_params[nms_ids, :2], 2), axis=1)) / np.sqrt(
            length_sqr[nms_ids]) * 2
    nonline_prob[nonline_prob > 1] = 1

    return nms_lines, nonline_prob


def generate_group_interpolate(lines: NDArray[np.int_]):
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
        xx: NDArray[np.int_] = np.ones(
            (num, ),
            dtype=np.int16) * line[0] if line[0] == line[2] else np.arange(
                line[0], line[2] + step_x, step=step_x).astype(np.int16)
        yy: NDArray[np.int_] = np.ones(
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


def map_list(func: Union[type, Callable[..., Any]],
             datalist: list[Any]) -> list[Any]:
    """Casting all elements in datalist with the given method.
    It can be a type or a callable function.

    Args:
        func (Union[type, Callable[..., Any]]): target data type.
        datalist (list[Any]): src data list

    Returns:
        list[Any]: datalist with all elements are converted.
    """
    return list(map(func, datalist))


def xywh2xyxy(mat: NDArray[np.float64],
              inplace: bool = True) -> NDArray[np.float64]:
    """Convert coordinates in format of (x,y,w,h) to (x1,y1,x2,y2).
    require multi-lines matrix with shape of (n,4).

    Args:
        mat (np.ndarray): input coordinate list.
        inplace (bool, optional): Whether generate results inplace. Defaults to True.

    Returns:
        np.ndarray: output coordinate list.
    """
    if inplace:
        mat[:, 0] = mat[:, 0] - mat[:, 2] / 2
        mat[:, 1] = mat[:, 1] - mat[:, 3] / 2
        mat[:, 2] = mat[:, 0] + mat[:, 2]
        mat[:, 3] = mat[:, 1] + mat[:, 3]
        return mat
    else:
        return np.array([
            mat[:, 0] - mat[:, 2] / 2, mat[:, 1] - mat[:, 3] / 2,
            mat[:, 0] + mat[:, 2], mat[:, 1] + mat[:, 3]
        ])


def met2xyxy(met: dict[str, list[int]]):
    """将met的字典转换为xyxy形式的坐标。

    Args:
        met (_type_): _description_
    """
    return Box.from_pts(met["pt1"], met["pt2"])


def calculate_area_iou(mat1: Box, mat2: Box):
    """用于计算面积的iou。

    Args:
        met_a (_type_): _description_
        met_b (_type_): _description_
    """
    # 若x/y轴宽为0，取另一轴做单轴比较
    if (mat1.x1 == mat1.x2 == mat2.x1 == mat2.x2):
        o_y = sorted([mat1.y1, mat1.y2, mat2.y1, mat2.y2], reverse=True)[1:-1]
        if mat1.y1 == mat2.y1 and mat1.y2 == mat2.y2:
            return 1
        return (o_y[2] - o_y[1]) / (o_y[3] - o_y[0])
    if (mat1.y1 == mat1.y2 == mat2.y1 == mat2.y2):
        o_x = sorted([mat1.x1, mat1.x2, mat2.x1, mat2.x2], reverse=True)[1:-1]
        if mat1.x1 == mat2.x1 and mat1.x2 == mat2.x2:
            return 1
        return (o_x[2] - o_x[1]) / (o_x[3] - o_x[0])

    # 若无交集即为0
    if (mat1.x1 >= mat2.x2 or mat1.x2 <= mat2.x1) or (mat1.y1 >= mat2.y2
                                                      or mat1.y2 <= mat2.y1):
        return 0

    # 计算交集面积
    i_xx = sorted([mat1.x1, mat1.x2, mat2.x1, mat2.x2], reverse=True)[1:-1]
    i_yy = sorted([mat1.y1, mat1.y2, mat2.y1, mat2.y2], reverse=True)[1:-1]
    area_i = (i_xx[1] - i_xx[0]) * (i_yy[1] - i_yy[0])

    # 分别计算面积
    area_a = (mat1.x2 - mat1.x1) * (mat1.y2 - mat1.y1)
    area_b = (mat2.x2 - mat2.x1) * (mat2.y2 - mat2.y1)
    return area_i / (area_a + area_b - area_i)


def box_matching(src_seq: Sequence[list[int]],
                 tgt_seq: Sequence[list[int]],
                 iou_threshold: float = 0.5):
    """box matching by iou. create idx from src2tgt.
    Args:
        src_box (_type_): _description_
        tgt_box (_type_): _description_
    """
    match_ind: list[tuple[int, int]] = []
    matched_tgt: list[int] = []
    tgt_boxes = [Box.from_list(x) for x in tgt_seq]
    src_boxes = [Box.from_list(x) for x in src_seq]
    for i, src_box in enumerate(src_boxes):
        best_iou, best_ind = 0, -1
        for j, tgt_box in enumerate(tgt_boxes):
            if j in matched_tgt: continue
            iou = calculate_area_iou(src_box, tgt_box)
            if iou > best_iou:
                best_iou = iou
                best_ind = j
        if best_ind != -1:
            match_ind.append((i, best_ind))
            matched_tgt.append(best_ind)
    return match_ind


def relative2abs_path(rpath: str):
    """Convert a relative path to the corresponding absolute path.

    Args:
        path (str): ./relative/path

    Returns:
        str: /the/absolute/path
    """
    if rpath.startswith("./"):
        rpath = rpath[2:]
    return path.join(WORK_PATH, rpath)


def expand_cls_pred(cls_pred: NDArray[np.float64]) -> NDArray[np.float64]:
    """expand cls prediction from [num, cls] to [num, cls+1].

    Args:
        cls_pred (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    num_pred, _ = cls_pred.shape
    return np.concatenate([cls_pred, np.zeros((num_pred, 1))], axis=-1)


def mod_all_attrs_to_cfg(cfg: EasyDict, name: str, action: str,
                         kwargs: dict[str, Any]) -> EasyDict:
    """ 修改cfg中的对应属性。

    Args:
        cfg (EasyDict): _description_
        name (str): _description_
        action (str): _description_
    """
    # currently only "add" action is supported.
    assert action in [
        "add",
    ], f"action {action} is not supported!"
    for key in cfg.keys():
        if isinstance(cfg[key], EasyDict):
            mod_all_attrs_to_cfg(cfg[key], name, action, kwargs)
        if key == name:
            if action == "add":
                for (new_key, new_attr) in kwargs.items():
                    cfg[key][new_key] = new_attr
    return cfg


ID2NAME: dict[int, str] = {}
NAME2ID: dict[str, int] = {}
with open(relative2abs_path("./global/class_name.txt")) as f:
    mapper = [x.strip().split() for x in f.readlines()]
    for num, name in mapper:
        ID2NAME[int(num)] = name
        NAME2ID[name] = int(num)
MAX_EXISTING_ID = max(ID2NAME.keys())
ID2NAME[MAX_EXISTING_ID + 1] = "DROPPED"
ID2NAME[MAX_EXISTING_ID + 2] = "OTHERS"
NAME2ID["DROPPED"] = MAX_EXISTING_ID + 1
NAME2ID["OTHERS"] = MAX_EXISTING_ID + 2
# NUM_CLASS here includes "OTHERS" with the last label
NUM_CLASS = len(ID2NAME)
