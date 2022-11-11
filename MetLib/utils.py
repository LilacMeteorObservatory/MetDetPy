import sys
from functools import partial
import warnings
import cv2
import numpy as np

from .VideoLoader import ThreadVideoReader


class Munch(object):

    def __init__(self, idict) -> None:
        for (key, value) in idict.items():
            #if isinstance(value,dict):
            #    value = Munch(value)
            setattr(self, key, value)


def sigma_clip_average(sequence, sigma=3.00):
    mean, std = np.mean(sequence), np.std(sequence)
    while True:
        # update sequence
        sequence = [x for x in sequence if abs(mean - x) < sigma * std]
        updated_mean, updated_std = np.mean(sequence), np.std(sequence)
        if updated_mean == mean:
            return updated_mean
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
    video = cv2.VideoCapture(video_name)
    if (video is None) or (not video.isOpened()):
        raise FileNotFoundError(
            "The file \"%s\" cannot be opened as a supported video format." %
            video_name)
    raw_wh = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    resize_param = parse_resize_param(resize_param, raw_wh)
    return video, load_mask(mask_name, resize_param) if mask_name else np.ones(
        (resize_param[1], resize_param[0]), dtype=np.uint8)


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


def set_out_pipe(workmode):
    if workmode == "backend":
        return stdout_backend
    elif workmode == "frontend":
        return print


def stdout_backend(*args):
    print(*args)
    sys.stdout.flush()


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


def _rf_est_kernel(video,
                   n_frames,
                   img_mask,
                   start_frame=0,
                   resize_param=None):
    try:
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        video_loader = ThreadVideoReader(
            video, n_frames,
            partial(preprocessing, mask=img_mask, resize_param=resize_param))
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
        rmax_pos = np.where((2 * A2 - (A1 + A3) > 0) & (2 * A1 - (A0 + A2) < 0)
                            & (np.abs(diff_series[1:-1]) > 0.01))[0]
        #plt.scatter(rmax_pos + 1, diff_series[rmax_pos + 1], s=30, c='r')
        #plt.plot(diff_series, 'r')
        #plt.show()
    finally:
        video_loader.stop()
    return rmax_pos[1:] - rmax_pos[:-1]


def rf_estimator(video, img_mask, resize_param):
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
        intervals = _rf_est_kernel(video, total_frame, img_mask, 0,
                                   resize_param)
    else:
        # 超过300帧的 从开头 中间 结尾各抽取100帧长度的视频进行估算。
        # TODO: 规避bug的设计 不太美观。
        intervals_1 = _rf_est_kernel(video, 100, img_mask, 0, resize_param)
        intervals_2 = _rf_est_kernel(video, 100, img_mask,
                                     (total_frame - 100) // 2, resize_param)
        intervals_3 = _rf_est_kernel(video, 100, img_mask, total_frame - 110,
                                     resize_param)
        intervals = np.concatenate([intervals_1, intervals_2, intervals_3])
    if len(intervals) == 0:
        return 1
    # 非常经验的取值方法...
    est_frames = np.round(min(np.median(intervals), sigma_clip_average(intervals)))
    return est_frames


def init_exp_time(exp_time, video, mask, resize_param, upper_bound=0.25):
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
        exp_time, (str, float, int)
    ), "exp_time should be either <str, float, int>, got %s" % (type(exp_time))
    if isinstance(exp_time, str):
        if exp_time == "real-time":
            return 1 / fps
        if exp_time == "slow":
            # TODO: Any better idea?
            return 1 / 4
        if exp_time == "auto":
            rf = rf_estimator(video, mask, resize_param)
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


def test_tf_estimator(test_video, test_mask, resize_param):
    video, img_mask = load_video_and_mask(test_video, test_mask, resize_param)
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
