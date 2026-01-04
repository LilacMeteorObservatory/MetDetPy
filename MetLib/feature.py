"""包含了用于计算图像或者序列特征的函数集合。"""

from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from .utils import U8Mat
from .metstruct import Box


def crop_with_box(img: U8Mat, roi: Box):
    return img[roi.y1:roi.y2, roi.x1:roi.x2]


def calc_roi_gradient(img: U8Mat, mask: Optional[NDArray[np.bool_]] = None):
    if len(img.shape) == 3 and img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dx_img = np.array(img, copy=True)
    dx_img[:-1] = dx_img[1:]
    dy_img = np.array(img, copy=True)
    dy_img[:, 1:] = dy_img[:, :-1]
    Ix = img.astype(np.int16) - dx_img
    Iy = img.astype(np.int16) - dy_img
    Ia = np.arctan2(Ix, Iy)
    weight = np.sqrt(Ix.astype(np.int64)**2 + Iy.astype(np.int64)**2)
    if mask:
        weight = weight * mask
    sum_weight = np.sum(weight)
    complex_sum = np.sum(weight * np.exp(1j * Ia))
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_complex = complex_sum / sum_weight
    mean_angle = np.angle(mean_complex)
    return mean_angle % (2 * np.pi)


def calc_brightness_with_roi(img: U8Mat,
                             roi: Optional[Box] = None,
                             gamma: float = 2.2) -> float:
    """
    calculate the brightness of the target in the image.
    
    If roi is provided, calculate the brightness with improved roi.
    (This avoid estimate the background with a large foreground,
    which may be inaccurate.);
    Otherwise, it calculates within the whole image.
    
    Args:
        img (U8Mat): the full-size input image.
        roi (Box): ROI box.
    """
    # TODO: this ignore the affects of bright stars. If the result is
    # inaccurate still, this should be fixed.
    if len(img.shape) == 3 and img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if roi is not None:
        (x, y), (w, h) = roi.to_xywh_list()
        long_side = max(w, h)
        rebuild_box = Box(max(0, x - long_side), max(0, y - long_side),
                          x + long_side, y + long_side)
        roi_img = crop_with_box(img, rebuild_box)
    else:
        roi_img = img
    blured_img = cv2.blur(roi_img, (5, 5))
    _, mask = cv2.threshold(blured_img, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bg_estimate = np.mean(blured_img[mask == 255])
    fg_estimate = np.mean(blured_img[mask == 0])
    return float(fg_estimate / bg_estimate)**(1 / gamma)
