"""计算图像或者序列特征的接口。"""
import cv2
import numpy as np
from .utils import U8Mat
from typing import Optional
from numpy.typing import NDArray

def calc_roi_direction(img: U8Mat):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    Ix = cv2.Sobel(img, cv2.CV_16S,1,0,3)
    Iy = cv2.Sobel(img, cv2.CV_16S,0,1,3)
    # 转换为角度
    weight = Ix.astype(np.int64)**2 + Iy.astype(np.int64)**2
    weight = weight/np.max(weight)

    Ia = np.arctan2(Ix, Iy)
    angle, unified, _ = weighted_mean_angle(Ia, weight)
    (angle / np.pi) * 360
    return angle, unified

    

def weighted_mean_angle(angle: NDArray[np.float_],
                        weight: Optional[np.float_] = None,
                        mask: Optional[np.bool_] = None):
    """
    返回 (mean_angle, R, sum_weights)：
    - mean_angle: 平均角度（弧度，-pi..pi）
    - R: 合向量长度归一化（0..1），R = |sum(w e^{iθ})| / sum(w)
    - sum_weights: 权重和（用于判断是否为 0）
    """
    a = angle
    w = weight if weight is not None else np.ones_like(a, dtype=np.float64)    
    if mask is not None:
        w = w * mask
    sum_w = np.sum(w)
    assert sum_w>0, 'sum of weight should >0'
    # 计算加权复数和
    complex_sum = np.sum(w * np.exp(1j * a))
    # 当权重和为 0 时避免除零
    # safe division
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_complex = complex_sum / sum_w
    # mean angle in [-pi, pi]
    mean_angle = np.angle(mean_complex)
    # resultant length R = |complex_sum| / sum_w
    R = np.abs(complex_sum) / sum_w
    # For zero-weight locations set sensible defaults
    return float(mean_angle), float(R), float(sum_w)

