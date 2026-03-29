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
    if img.ndim == 3 and img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # dI/dx
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # dI/dy
    Ia = np.arctan2(-gy, gx) % np.pi
    weight = np.hypot(gx, gy)
    if mask is not None:
        weight = weight * (mask.astype(weight.dtype))
    else:
        high_weight = np.percentile(weight, 95)
        weight = weight * (weight>high_weight)
    sum_weight = np.sum(weight)
    if sum_weight == 0:
        return float('nan')
    complex_sum = np.sum(weight * np.exp(1j * Ia))
    mean_angle = np.angle(complex_sum / sum_weight)
    return float(mean_angle % (2 * np.pi))


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
