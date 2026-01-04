"""包含了用于计算图像或者序列特征的函数集合。"""

from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from .utils import U8Mat


def rad2degree(rad: float):
    return (rad % (2 * np.pi)) * 180


def calc_roi_gradient(img: U8Mat, mask: Optional[NDArray[np.bool_]] = None):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Ix = cv2.Sobel(img, cv2.CV_16S, 1, 0, 3)
    Iy = cv2.Sobel(img, cv2.CV_16S, 0, 1, 3)
    Ia = np.arctan2(Iy, Ix) % np.pi
    weight = np.sqrt(Ix.astype(np.int64)**2 + Iy.astype(np.int64)**2)
    if mask:
        weight = weight * mask
    sum_weight = np.sum(weight)
    complex_sum = np.sum(weight * np.exp(1j * Ia))
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_complex = complex_sum / sum_weight
    mean_angle = np.angle(mean_complex)
    return rad2degree(mean_angle)