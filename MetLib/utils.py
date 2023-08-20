import datetime
import warnings
from functools import partial
from math import floor

import cv2
import numpy as np

from .VideoWarpper import OpenCVVideoWarpper
from .MetLog import get_default_logger

eps = 1e-2

img_max = partial(np.max, axis=0)
pt_len_xy = lambda pt1, pt2: (pt1[1] - pt2[1])**2 + (pt1[0] - pt2[0])**2
drct = lambda pts: np.arccos((pts[1][1] - pts[0][1]) /
                             (pt_len_xy(pts[0], pts[1]))**(1 / 2))
drct_line = lambda pts: np.arccos((pts[3] - pts[1]) /
                                  (pt_len_xy(pts[:2], pts[2:]))**(1 / 2))

logger = get_default_logger()


class EMA(object):
    """移动指数平均。
    可用于对平稳序列的评估。

    Args:
        object (_type_): _description_
    """

    def __init__(self, n) -> None:
        self.n = n
        self.ema_pool = []

    def update(self, num):
        self.ema_pool.append(num)
        self.ema_pool = self.ema_pool[-self.n:]

    @property
    def mean(self):
        return np.mean(self.ema_pool) if len(self.ema_pool) > 0 else np.inf

    @property
    def std(self):
        return np.std(self.ema_pool) if len(self.ema_pool) > 0 else np.inf


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


def parse_resize_param(tgt_wh, raw_wh):
    #TODO: fix poor English
    w, h = raw_wh
    if isinstance(tgt_wh, str):
        try:
            # if str, tgt_wh is from args.
            if ":" in tgt_wh or "x" in tgt_wh:
                if ":" in tgt_wh:
                    tgt_wh = list(map(int, tgt_wh.split(":")))
                else:
                    tgt_wh = list(map(int, tgt_wh.split("x")))
            else:
                tgt_wh = int(tgt_wh)
        except Exception as e:
            raise e("Unexpected values for argument \"--resize\".\
                 Input should be either one integer or two integers separated by \":\"(to be deprecated) or \"x\"."
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
    # OpenCVVideoWarpper is the default video capture source.
    # If you want to use your own way to load video, define your VideoWarpper
    # in MetLib/VideoWarpper.py and replace all OpenCVVideoWarpper in your code.
    video = OpenCVVideoWarpper(video_name)
    # Calculate real resize parameters.
    resize_param = parse_resize_param(resize_param, video.size)
    mask = None
    if mask_name:
        mask = load_mask(mask_name, resize_param)
    else:
        # If no mask_name is provided, use a all-pass mask.
        mask = np.ones((resize_param[1], resize_param[0]), dtype=np.uint8)
    return video, mask


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
            f"Unsupported suffix \"{suffix}\";\Only .png and .jpeg/.jpg are supported."
        )
    status, buf = cv2.imencode(ext, img, params)
    if status:
        with open(
                filename,
                mode='wb',
        ) as f:
            f.write(buf)
    else:
        raise Exception("imencode failed.")


def save_video(video_series, fps, video_path):
    try:
        real_size = list(reversed(video_series[0].shape[:2]))
        cv_writer = cv2.VideoWriter(video_path,
                                    cv2.VideoWriter_fourcc(*"MJPG"), fps,
                                    real_size)
        for clip in video_series:
            p = cv_writer.write(clip)
    finally:
        cv_writer.release()


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


def _rf_est_kernel(video_loader, n_frames):
    try:
        video_loader.start()
        f_sum = np.zeros((n_frames, ), dtype=float)
        for i in range(n_frames):
            if not video_loader.stopped:
                frame = video_loader.pop()
                f_sum[i] = np.sum(frame)
            else:
                f_sum = f_sum[:i]
                break
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


def rf_estimator(video_loader):
    """用于为给定的视频估算实际的曝光时间。

    部分相机在录制给定帧率的视频时，可以选择慢于帧率的单帧曝光时间（慢门）。
    还原真实的单帧曝光时间可帮助更好的检测。
    但目前没有做到很好的估计。

    Args:
        video (cv2.VideoCapture): 给定视频片段。
        mask (ndarray): the mask for the video.
    """
    total_frame = video_loader.iterations
    start_frame = video_loader.start_frame
    if total_frame < 300:
        # 若不超过300帧 则进行全局估算
        intervals = _rf_est_kernel(video_loader, total_frame)
    else:
        # 超过300帧的 从开头 中间 结尾各抽取100帧长度的视频进行估算。
        video_loader.reset(iterations=100)
        intervals_1 = _rf_est_kernel(video_loader, 100)

        video_loader.reset(start_frame=start_frame + (total_frame - 100) // 2,
                           iterations=100)
        intervals_2 = _rf_est_kernel(video_loader, 100)

        video_loader.reset(start_frame=start_frame + total_frame - 100,
                           iterations=100)
        intervals_3 = _rf_est_kernel(video_loader, 100)

        intervals = np.concatenate([intervals_1, intervals_2, intervals_3])
    if len(intervals) == 0:
        return 1
    # 非常经验的取值方法...
    est_frames = np.round(
        min(np.median(intervals), np.mean(sigma_clip(intervals))))
    return est_frames


def init_exp_time(exp_time, video_loader, upper_bound):
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
    fps = video_loader.video.fps
    logger.debug(f"Metainfo FPS = {fps:.2f}")
    assert isinstance(
        exp_time, (str, float, int)
    ), "exp_time should be either <str, float, int>, got %s" % (type(exp_time))

    if fps <= int(1 / upper_bound):
        logger.warning(f"Slow FPS detected. Use {1/fps:.2f}s directly.")
        return 1 / fps

    if isinstance(exp_time, str):
        if exp_time == "real-time":
            return 1 / fps
        if exp_time == "slow":
            # TODO: Any better idea?
            return 1 / 4
        if exp_time == "auto":
            rf = rf_estimator(video_loader)
            if rf / fps >= upper_bound:
                logger.warning(
                    f"Unexpected exposuring time (too long):{rf/fps:.2f}s. Use {upper_bound:.2f}s instead."
                )
            return min(rf / fps, upper_bound)
        try:
            exp_time = float(exp_time)
        except ValueError as E:
            raise ValueError(
                "Invalid exp_time string value: It should be selected from [float], [int], "
                + "real-time\",\"auto\" and \"slow\", got %s." % (exp_time))
    if isinstance(exp_time, (float, int)):
        if exp_time * fps < 1:
            logger.warning(
                f"Invalid exposuring time (too short). Use {1/fps:.2f}s instead."
            )
            return 1 / fps
        return float(exp_time)


def frame2ts(frame, fps):
    return datetime.datetime.strftime(
        datetime.datetime.utcfromtimestamp(frame / fps), "%H:%M:%S.%f")[:-3]

def time2frame(time: str, fps: float) -> int:
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
        time2frame("00:00:02.56",25) -> 64(=(2+(56/100))*25))
    """
    assert time.count(":")==2, f"Invaild time string: \":\" in \"{time}\" should appear exactly 2 times."
    if "." in time:
        dt = datetime.datetime.strptime(time, "%H:%M:%S.%f")
    else:
        dt = datetime.datetime.strptime(time, "%H:%M:%S")
    dt_time = dt.hour * 60**2 + dt.minute * 60**1 + dt.second + dt.microsecond / 1e6
    return int(dt_time * fps)

def timestr2int(time: str) -> int:
    """A wrapper of `time2frame`, is mainly used to turn time-string to its corresponding integer in ms.
    
    It supports input like:
        'NoneType' -> return 'NoneType' as well.
        
        `HH:MM:SS` or `HH:MM:SS.MS` -> return its corresponding integer in ms.
        
        A string that describes the time in ms -> its corresponding integer.

    Args:
        time (str): time string.

    Returns:
        int: corresponding integer in ms.
    """
    if time==None:
        return None
    if ":" in time:
        return time2frame(time, fps=1000)
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
        i = max(int((x - eps) / gap), 0)
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
    return min(std1, std2)


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
            cc = np.mean((sin_lines[:, :2] + sin_lines[:, 2:]) / 2, axis=0).astype(np.int16)
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
    coord_list = [None for i in range(len(lines))]
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


def least_square_fit(pts):
    """fit pts to the linear func Ax+By+C=0

    Args:
        pts (_type_): _description_

    Returns:
        tuple: parameters (A,B,C)
        float: average loss
    """
    pts = np.array(pts)
    avg_xy = np.mean(pts, axis=0)
    dpts = pts - avg_xy
    dxx, dyy = np.sum(dpts**2, axis=0)
    dxy = np.sum(dpts[:, 0] * dpts[:, 1])
    v, m = np.linalg.eig([[dxx, dxy], [dxy, dyy]])
    A, B = m[np.argmin(v)]
    # fixed
    # TODO: this works fine now. but why??
    B = -B

    C = -np.sum(np.array([A, B]) * avg_xy)
    loss = np.mean(np.abs(np.einsum("a,ba->b", np.array([A, B]), pts) + C))
    return (A, B, C), loss