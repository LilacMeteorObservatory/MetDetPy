import sys
from functools import partial

import cv2
import numpy as np

from .VideoLoader import ThreadVideoReader


def load_video_and_mask(video_name, mask_name=None, resize_param=(0, 0)):
    return cv2.VideoCapture(video_name), load_mask(
        mask_name, resize_param) if mask_name else np.ones(
            (resize_param[1], resize_param[0]), dtype=np.uint8)


def load_mask(filename, resize_param):
    mask = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), 1)
    mask = preprocessing(mask, resize_param=resize_param)
    return cv2.threshold(mask, 128, 1, cv2.THRESH_BINARY)[-1]


def preprocessing(frame, mask=1, resize_param=(0, 0)):
    frame = cv2.cvtColor(cv2.resize(frame, resize_param), cv2.COLOR_BGR2GRAY)
    return frame * mask


def set_out_pipe(workmode):
    if workmode == "backend":
        return stdout_backend
    elif workmode == "frontend":
        return print


def stdout_backend(*args):
    print(*args)
    sys.stdout.flush()


def m3func(image_stack, skipping=1):
    """M3 for Max Minus Median.
    Args:
        image_stack (ndarray)
    """
    sort_stack = np.sort(image_stack[::skipping], axis=0)
    return sort_stack[-1] - sort_stack[len(sort_stack) // 2]


def _rf_est_kernel(video,
                   n_frames,
                   img_mask,
                   start_frame=0,
                   resize_param=(960, 540)):
    try:
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        video_loader = ThreadVideoReader(video, n_frames,
                                         partial(
                                             preprocessing,
                                             mask=img_mask,
                                             resize_param=resize_param))
        reg = resize_param[0] * resize_param[1]
        video_loader.start()
        f_sum = np.zeros((n_frames, ), dtype=np.float)
        for i in range(n_frames):
            f_sum[i] = np.sum(video_loader.pop(1)[0] / reg)
        # mean filter with windows_size=3
        #f_sum = np.array([(
        #    f_sum[max(0, i - 1)] + f_sum[i] + f_sum[min(len(f_sum) - 1, i + 1)]
        #) / 3 for i, _ in enumerate(f_sum)])

        A0, A1, A2, A3 = f_sum[:-3], f_sum[1:-2], f_sum[2:-1], f_sum[3:]

        diff_series = f_sum[1:] - f_sum[:-1]
        rmax_pos = np.where((2 * A2 - (A1 + A3) > 0) & (
            2 * A1 - (A0 + A2) < 0) & (np.abs(diff_series[1:-1]) > 0.01))[0]
        #plt.scatter(rmax_pos + 1, diff_series[rmax_pos + 1], s=30, c='r')
        #plt.plot(diff_series, 'r')
        #plt.show()
    finally:
        video_loader.stop()
    return rmax_pos[1:] - rmax_pos[:-1]


def rf_estimator(video, img_mask):
    """用于为给定的视频估算实际的曝光时间。

    部分相机在录制给定帧率的视频时，可以选择慢于帧率的单帧曝光时间（慢门）。
    还原真实的单帧曝光时间可帮助更好的检测。
    但目前没有做到很好的估计。

    Args:
        video (cv2.VideoCapture): 给定视频片段。
        mask (ndarray): the mask for the video.
    """
    total_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frame < 300:
        # 若不超过300帧 则进行全局估算
        intervals = _rf_est_kernel(video, total_frame, img_mask, 0)
    else:
        # 超过300帧的 从开头 中间 结尾各抽取100帧长度的视频进行估算。
        intervals_1 = _rf_est_kernel(video, 100, img_mask, 0)
        intervals_2 = _rf_est_kernel(video, 100, img_mask,
                                     (total_frame - 100) // 2)
        intervals_3 = _rf_est_kernel(video, 100, img_mask, total_frame - 101)
        intervals = np.concatenate([intervals_1, intervals_2, intervals_3])
    est_frames = np.median(intervals)
    return est_frames


def init_exp_time(exp_time, video, mask):
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
    fps = video.get(cv2.CAP_PROP_FPS)
    assert isinstance(
        exp_time,
        (str, float,
         int)), "exp_time should be either <str, float, int>, got %s" % (
             type(exp_time))
    if isinstance(exp_time, (float, int)):
        return float(exp_time)
    else:
        if exp_time == "real-time":
            return 1 / fps
        if exp_time == "slow":
            # TODO: Any better idea?
            return 1 / 4
        if exp_time == "auto":
            rf = rf_estimator(video, mask)
            return rf / fps
        raise ValueError(
            "Invalid exp_time string value. It should be selected from \
                \"real-time\",\"auto\" and \"slow\", got %s." % (exp_time))


def test_tf_estimator(test_video, test_mask):
    video, img_mask = load_video_and_mask(
        test_video, test_mask, resize_param=(960, 540))
    assume_rf = rf_estimator(video, img_mask)
    print(
        "The given video is assumed to have a exposure time of %.2f s each frame."
        % (assume_rf / video.get(cv2.CAP_PROP_FPS)))


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
            des[k, position], des[k + 1, position] = des[k + 1, position], des[
                k, position]
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
    diff_img = np.reshape(window_stack[sort_stack[-1],
                                       np.where(sort_stack[-1] >= 0)[0]],
                          (H, W))
    # 形状还原
    window_stack = np.reshape(window_stack, (L, H, W))
    sort_stack = np.reshape(sort_stack, (L, H, W))

    #update and calculate
    window_stack.append(frame)
    if len(window_stack) > window_size:
        window_stack.pop(0)
    diff_img = np.max(window_stack, axis=0) - np.median(window_stack, axis=0)
    return diff_img
