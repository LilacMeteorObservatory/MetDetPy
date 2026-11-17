"""
包含用于图像处理的各种函数。
"""

from typing import Any, Callable, NamedTuple, Optional, Sequence, Union

import cv2
import numpy as np
from numpy.typing import NDArray

from .profiling import profiled
from .utils import U8Mat, keep1ret_value

UINT8_MAX = 255
UINT16_MAX = 65535


def scale2tgt_mean(img_lin: NDArray[np.uint16],
                   power: float = 2.222,
                   target_nl_mean: float = 0.3) -> NDArray[np.uint16]:
    """Compute multiplier k so that `mean( (slope * (k*L)) ** (1/power) ) == target_nl_mean`,
    assuming `f(L) = (slope * L) ** (1/power)` and `L` is linear luminance.

    Args:
        img_lin (NDArray[np.uint16]): linear input image
        power (float, optional): gamma transforamtion base value. Defaults to 2.222. (used as 1/power)
        target_nl_mean (float, optional): target luminance average value. Defaults to 0.3.

    Returns:
        NDArray[np.uint16]: scaled non-linear image
    """
    l_gray_mean = np.mean(cv2.cvtColor(img_lin, cv2.COLOR_BGR2GRAY))
    k = (target_nl_mean**power) / l_gray_mean
    return ((k * img_lin)**(1 / power) * UINT16_MAX).clip(
        0, UINT16_MAX).astype(np.uint16)


def contrast_stretch_uint16(img_uint16: NDArray[np.uint16],
                            alpha: float = 1.2) -> NDArray[np.uint16]:
    # 对比度拉伸
    lab_img = cv2.cvtColor((img_uint16 / UINT16_MAX).astype(np.float32),
                           cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab_img)
    L_mean = L.mean()
    L_scaled = (L - L_mean) * alpha + L_mean
    L_scaled = np.clip(L_scaled, 0.0, 100.0)
    lab_scaled = cv2.merge([L_scaled, a, b])
    rgb_scaled = cv2.cvtColor(lab_scaled, cv2.COLOR_LAB2BGR)
    return (rgb_scaled * UINT16_MAX).astype(np.uint16)


def contrast_stretch_uint8(img_uint8: U8Mat, alpha: float = 1.2) -> U8Mat:
    """ Apply contrast stretching in LAB color space for U8Mat.

    Args:
        img_uint8 (U8Mat): input image in uint8 format.
        alpha (float, optional): contrast stretching factor. Defaults to 1.2.

    Returns:
        U8Mat: contrast-stretched image in uint8 format.
    """
    lab_img = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab_img)
    L_mean = L.mean()
    L_scaled = (L - L_mean) * alpha + L_mean
    L_scaled = np.clip(L_scaled, 0, UINT8_MAX).astype(np.uint8)
    rgb_scaled = cv2.cvtColor(cv2.merge([L_scaled, a, b]), cv2.COLOR_LAB2BGR)
    return rgb_scaled


class Size(NamedTuple):
    """ROI/模型输入尺寸，字段名显式，避免 (w, h) 与 (h, w) 的顺序歧义。

    它是 `tuple` 的子类，因此可以直接传给 `cv2.resize(size)`；需要 numpy 形状时用
    ``(size.height, size.width)`` 显式取值。
    """

    width: int
    height: int

    def as_hw(self) -> tuple[int, int]:
        """返回 numpy/ONNX 风格 ``(height, width)``。"""
        return self.height, self.width

    def as_wh(self) -> tuple[int, int]:
        """返回 OpenCV 风格 ``(width, height)``。"""
        return self.width, self.height


def normalize_size(size: Union[int, Sequence[int]]) -> Size:
    """把"尺寸"统一成 `Size`（OpenCV 风格：标量表示正方形）。

    Args:
        size (Union[int, Sequence[int]]): 标量边长，或长度 2 的序列 ``(w, h)``。

    Returns:
        Size: 归一化后的尺寸。

    Raises:
        ValueError: 尺寸不是正整数，或序列长度不是 2。
    """
    if isinstance(size, (int, np.integer)):
        values = (int(size), int(size))
    elif isinstance(size, (str, bytes)) or not hasattr(size, '__len__'):
        raise ValueError(
            f"ROI size must be an int or a (width, height) pair, got {size!r}."
        )
    else:
        if len(size) != 2:
            raise ValueError(
                f"ROI size must be an int or a (width, height) pair, got {size!r}."
            )
        values = (int(size[0]), int(size[1]))
    if values[0] <= 0 or values[1] <= 0:
        raise ValueError(f"ROI size must be positive, got {values!r}.")
    return Size(width=values[0], height=values[1])


def plan_roi_size(width: float,
                  height: float,
                  canvas: Union[int, Sequence[int]],
                  min_side: int = 224,
                  scale: float = 1.2,
                  target_long_side: Optional[int] = None,
                  side_floor: Optional[int] = None) -> Size:
    """规划 ROI 裁切尺寸：默认"1.2× 框 + 最小边长"，可选按目标占比自适应。

    裁切宽高比与 `canvas` 一致（缩放到输入时不会拉伸）；`canvas` 给标量时退化为正方形
    裁切，与历史公式逐值一致。

    Args:
        width (float): 框宽度（像素）。
        height (float): 框高度（像素）。
        canvas (Union[int, Sequence[int]]): 模型输入尺寸；标量=正方形，``(w, h)``=长方形。
        min_side (int): 裁切**长边**下限，同时也是自适应的上界（不裁得比它更大）。
        scale (float): 相对框的放大倍数（默认 1.2，与训练侧裁切规则一致）。
        target_long_side (Optional[int]): 自适应目标：让目标长边在输入画布的长轴上占这么多
            像素。None 表示保持历史行为（只用 min_side + scale）。
        side_floor (Optional[int]): 自适应时裁切长边的下限，用于限制最大放大倍数。

    Returns:
        Size: 裁切尺寸（``width``/``height`` 均不小于 1）。
    """
    canvas_size = normalize_size(canvas) if not isinstance(canvas,
                                                           Size) else canvas
    canvas_w, canvas_h = canvas_size.as_wh()
    canvas_long = float(max(canvas_w, canvas_h))
    aspect = min(canvas_w, canvas_h) / canvas_long
    box_long = max(float(width), float(height), 1.0)

    if target_long_side:
        desired = int(np.ceil(canvas_long * box_long /
                              float(target_long_side)))
        floor = int(side_floor) if side_floor else 1
        crop_long = min(max(desired, floor), int(min_side))
    else:
        crop_long = int(min_side)

    # 三条约束取最大：紧裁倍数、短轴方向也要装得下框、画布长宽比。
    crop_long = max(crop_long, int(np.ceil(float(scale) * box_long)),
                    int(np.ceil(float(scale) * min(width, height) / aspect)))
    crop_long = max(1, crop_long)
    crop_short = max(1, int(round(crop_long * aspect)))
    if canvas_w >= canvas_h:
        return Size(width=crop_long, height=crop_short)
    return Size(width=crop_short, height=crop_long)


def crop_roi(img: U8Mat,
             pt1: list[int],
             pt2: list[int],
             size: Union[int, Sequence[int]],
             pad_value: int = 0) -> U8Mat:
    """从图像中裁出一个以框中心为中心、尺寸为 `size` 的矩形区域。

    Args:
        img (U8Mat): 源图（HWC）。
        pt1 (list[int]): 框的一角。
        pt2 (list[int]): 框的另一角。
        size (Union[int, Sequence[int]]): 裁切尺寸；标量=正方形，``(w, h)``=长方形（内部按 `Size` 处理）。
        pad_value (int): 越界填充值。

    Returns:
        U8Mat: ``height × width`` 的 ROI（越界部分补 `pad_value`）。
    """
    crop_size = normalize_size(size) if not isinstance(size, Size) else size
    crop_w, crop_h = crop_size.as_wh()
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    center_x = (min(x1, x2) + max(x1, x2)) / 2.0
    center_y = (min(y1, y2) + max(y1, y2)) / 2.0
    roi_left = int(np.floor(center_x - crop_w / 2.0))
    roi_top = int(np.floor(center_y - crop_h / 2.0))
    roi = np.full((crop_h, crop_w, img.shape[2]), pad_value, dtype=img.dtype)
    src_x1, src_y1 = max(0, roi_left), max(0, roi_top)
    src_x2 = min(img.shape[1], roi_left + crop_w)
    src_y2 = min(img.shape[0], roi_top + crop_h)
    if src_x2 > src_x1 and src_y2 > src_y1:
        roi[src_y1 - roi_top:src_y2 - roi_top,
            src_x1 - roi_left:src_x2 - roi_left] = img[src_y1:src_y2,
                                                       src_x1:src_x2]
    return roi


class Transform(object):
    """图像变换方法的集合类，及一个用于执行集成变换的方法。
    """
    MASK_FLAG = "MASK"
    PATTERN_MAPPING = {
        "BGGR": cv2.COLOR_BAYER_BGGR2BGR,
        "RGGB": cv2.COLOR_BAYER_RGGB2BGR
    }

    def __init__(self) -> None:
        self.transform: list[tuple[Callable[..., U8Mat], dict[str, Any]]] = []

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

    def mask_with(self, mask: U8Mat):

        def _mask_with(img: U8Mat, mask: U8Mat):
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

    def tap(self, callback: Callable[[U8Mat], U8Mat]):
        """插入一个只观察、不改变图像的步骤（用于旁路输出中间产物）。

        典型的用法是插在 resize 之后、灰度转换之前，把 runtime 尺寸的彩色帧
        交给复核帧缓存 —— 这一步没有额外转换成本，只是保留已经存在的数组。

        Args:
            callback (Callable[[U8Mat], U8Mat]): 接收当前中间图像，返回用于
                继续流水线的图像（通常原样返回）。
        """
        self.transform.append((callback, dict()))

    @profiled("preprocess.total")
    def exec_transform(self, img: U8Mat) -> U8Mat:
        """按顺序对给定的输入执行给定的图像变换。

        Args:
            img (U8Mat): 输入图像
        Returns:
            MatLike: 变换后图像
        """
        for (transform, kwargs) in self.transform:
            img = transform(img, **kwargs)
        return img

    def scale2tgt_mean(self,
                       power: float = 2.222,
                       target_nl_mean: float = 0.3):
        self.transform.append(
            (scale2tgt_mean, dict(power=power, target_nl_mean=target_nl_mean)))

    def contrast_stretch_uint16(self, alpha: float = 1.2):
        self.transform.append((contrast_stretch_uint16, dict(alpha=alpha)))

    def contrast_stretch_uint8(self, alpha: float = 1.2):
        self.transform.append((contrast_stretch_uint8, dict(alpha=alpha)))

    def u16_to_u8(self):

        def _u16_to_u8(img: NDArray[np.uint16]) -> U8Mat:
            return (img // 257).astype(np.uint8)

        self.transform.append((_u16_to_u8, dict()))
