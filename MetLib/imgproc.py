"""
包含用于图像处理的各种函数。
"""

from typing import Any, Callable, Union

import cv2
import numpy as np
from numpy.typing import NDArray

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

    def exec_transform(self, img: U8Mat) -> U8Mat:
        """按顺序对给定的输入执行给定的图像变换。

        Args:
            img (U8Mat): 输入图像
        Returns:
            MatLike: 变换后图像
        """
        for [transform, kwargs] in self.transform:
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
