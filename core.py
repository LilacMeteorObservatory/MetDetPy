#import logging
import argparse
import json
import sys
import time

import cv2
import numpy as np
import tqdm
#import asyncio
import threading

from meteorlib import MeteorCollector

## baseline:
## 42 fps; tp 4/4 ; tn 0/6 ; fp 0/8.

## spring-beta: 11 fps(debug mode);tp 4/4 (some not believeable); tn 3/6 (2 of which are REALLY-HARD!); fp 4/8.(Why???)
## spring-v1: 9 fps (debug mode)/ 16 fps (speed mode); tp 4/4; tn 4/6; fp 8/11.
## spring-v2: 20 fps (no-skipping); 25 fps(median-skipping, fp 6/8)

pi = 3.141592653589793 / 180.0

progout = None

# POSITIVE: 3.85 3.11 3.03 2.68 2.55 2.13 2.61 1.94
# NEGATIVE: 0.49  0.65 2.96 5.08 2.44  1.49 2.69 7.52 19.45 11.18 13.96

def output_meteors(update_info):
    met_lst,drop_lst = update_info
    for met in met_lst:
        progout("Meteor: %s"%met)
    #for met in drop_lst:
    #    progout("Dropped: %s"%met)


def set_out_pipe(workmode):
    global progout
    if workmode == "backend":
        progout = stdout_backend
    elif workmode == "frontend":
        progout = print


def stdout_backend(string: str):
    sys.stdout.write(string)
    sys.stdout.flush()


def DrawHist(src, mask, hist_num=256, threshold=0):
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([src], [0], mask, [hist_num],
                        [0, hist_num - 1])[:, 0] / np.sum(mask)
    h, w = src.shape
    bw = w / hist_num
    canvas = np.zeros_like(src, dtype=np.uint8)
    i = threshold
    canvas = cv2.rectangle(canvas, (int(bw * i), h), (int(bw * i), 0),
                           [255, 255, 255], 2, -1)
    for i, bar in enumerate(hist):
        canvas = cv2.rectangle(canvas, (int(bw * i), h),
                               (int(bw * (i + 1)), h - int(bar * h)),
                               [128, 128, 128], 2, -1)
    return canvas


#@numba.jit(nopython=True, fastmath=True, parallel=True)
def sanaas(stack: np.array, des: np.array, boolmap, L, H, W):
    '''
    ================================DEPRECATED========================
    stack [list[np.array]],
    new_matrix H*W
    des[list[np.array]] L*H*W stack[des]是真实升序序列。des是下标序列。
    已实现可借助numba的加速版本；但令人悲伤的仍然是其运行速度。
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


def GammaCorrection(src, gamma):
    """deprecated."""
    return np.array((src / 255.)**(1 / gamma) * 255, dtype=np.uint8)


def load_video_mask(video_name, mask_name=None, resize_param=(0, 0)):
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


def detect_within_window(diff_img: np.array,
                         cfg: dict,
                         drawing=None,
                         mask=None,
                         visual_param=None,
                         debug_mode=False) -> tuple:
    """Detector, but version spring.

    主要工作原理： 以X帧为窗口的帧差法 （最大值-中值）。
    
    采取了相比原算法更激进的阈值和直线检测限，将假阳性样本通过排除记录流星的算法移除。

    Args:
        stack (_type_): _description_
        threshold (_type_): _description_
        drawing (_type_): _description_
        debug_mode (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    bi_threshold, line_threshold, line_minlen, median_skipping = cfg[
        "bi_threshold"], cfg["line_threshold"], cfg["line_minlen"], cfg[
            "median_skipping"]
    # 初始时不进行中位数跳采估算
    #if len(stack)<=median_skipping*(cfg["median_sampling_num"]-1):
    #    median_skipping=1
    _, dst = cv2.threshold(diff_img, bi_threshold, 255, cv2.THRESH_BINARY)
    linesp = cv2.HoughLinesP(
        np.array(dst, dtype=np.uint8), 1, pi, line_threshold, line_minlen, 0)

    if debug_mode:
        diff_img = cv2.cvtColor(
            np.array(diff_img, np.uint8), cv2.COLOR_GRAY2BGR)
        drawing = np.repeat(np.expand_dims(drawing, axis=-1), 3, axis=-1)
        y, x = visual_param
        canvas = np.zeros((x * 2, y * 2, 3), dtype=np.uint8)
        canvas[:x, :y] = cv2.resize(drawing, (y, x))
        canvas[:x, y:] = cv2.resize(diff_img, (y, x))
        canvas[x:, :y] = cv2.cvtColor(
            cv2.resize(
                DrawHist(diff_img, mask, threshold=bi_threshold), (y, x)),
            cv2.COLOR_GRAY2BGR)
        canvas[x:, y:] = cv2.resize(drawing, (y, x))
        drawing = canvas

    if linesp is None:
        return False, [], drawing
    return True, linesp[0], drawing


def detect_within_window_raw(stack, x, drawing, y, debug_mode=False):
    # 来自日本人版本的检测器。
    # 为统一API，在接口增加了不会使用的x,y。
    # 4帧为窗口，差分2,3帧，二值化，膨胀（高亮为有差异部分）
    #drawing = np.repeat(np.expand_dims(drawing, axis=-1), 3, axis=-1)
    diff23 = cv2.absdiff(stack[2], stack[3])
    _, diff23 = cv2.threshold(diff23, 20, 255, cv2.THRESH_BINARY)

    diff23 = cv2.dilate(
        diff23,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    diff23 = 255 - diff23
    ## 用diff23和0,1帧做位与运算（掩模？），屏蔽2,3帧的差一部分
    f1 = cv2.bitwise_and(diff23, stack[0])
    f2 = cv2.bitwise_and(diff23, stack[1])
    ## 差分0,1帧，二值化，膨胀（高亮有差异部分）
    dst = cv2.absdiff(f1, f2)
    _, dst = cv2.threshold(dst, 20, 255, cv2.THRESH_BINARY)
    dst = cv2.dilate(
        dst,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )

    # 对0,1帧直线检测（即：在屏蔽了2,3帧变化的图上直线检测。为毛？）
    # 所以即使检出应该也是第一帧上面检出。
    linesp = cv2.HoughLinesP(dst, 1, pi, 10, 10, 0)
    if not (linesp is None):
        linesp = linesp[0]
        #progout(linesp)
        for pt in linesp:
            progout("center=(%.2f,%.2f)" % ((pt[3] + pt[1]) / 2,
                                            (pt[2] + pt[0]) / 2))
            #cv2.line(drawing, (pt[0], pt[1]), (pt[2], pt[3]), (0, 0, 255), 10)
    return drawing, _, _


#async def video_loader_generator(video,iterations,mask,resize_param):
#    for i in range(iterations):
#        status, frame = video.read()
#        if not status:
#            break
#        frame = preprocessing(frame, mask=mask, resize_param=resize_param)
#        yield frame
'''
# 配置参数
# 常规设置
# resize_param ： 描述代码主干在何种分辨率下进行检测。更低的分辨率可以做到更高的fps，更高的则可能检测到更暗弱和短的流星。
# visual_param ： Debug模式下使用何种分辨率进行可视化。不启用Debug模式时，无需配置该项。
# window_size ： 时间滑窗大小（以秒为单位），程序将自动将其转换为帧数。建议使用默认值。
resize_param = (960, 540)
visual_param = (480, 270)
window_size_s = 0.36

# 流星检测参数

# bi_threshold ：描述检出流星所使用的阈值。可以根据使用的ISO进行调整，过低可能会引入更多噪声误检。
# median_sampling_num ：描述中位数的采样数目。更少的采样数目可能会引发低信噪比导致的误差，但可以达到更高的速度。设置-1时表示不跳采。
# line_* : 直线检测参数。默认情况下不用动。

detect_cfg=dict(
    bi_threshold = 5.4,
    median_sampling_num=-1,
    line_threshold=10,
    line_minlen=16)

# 流星判别参数

# min_len ：开始记录流星所需要的最小长度（占长边的比）。
# max_interval：流星最长间隔时间（经过该时间长度没有响应的流星将被认为已经结束）。单位：s。
# time_range ： 描述流星的持续时间范围。超过或者没有到达阈值的响应将会被排除。单位：s。
# speed_range ： 描述流星的速度范围。超过或者没有到达阈值的响应将会被排除。单位：frame^(-1)。
# thre ： 描述若干响应之间允许的最长距离平方。

meteor_cfg_inp = dict(
    min_len=10,
    max_interval=4,
    time_range=(0.12, 10),
    speed_range=(1.6, 4.6),
    thre2=320)
'''


class VideoRead(object):
    def __init__(self, video, iterations, mask, resize_param) -> None:
        self.video = video
        self.iterations = iterations
        self.mask = mask
        self.resize_param = resize_param
        self.stopped = False
        self.status = False
        self.frame_pool = []
        self.load_a_frame()

    def start(self):
        self.thread = threading.Thread(target=self.get, args=())
        self.thread.start()
        return self

    def load_a_frame(self):
        self.status, frame = self.video.read()
        if self.status:
            self.frame = preprocessing(
                frame, mask=self.mask, resize_param=self.resize_param)
            self.frame_pool.append(self.frame)
        else:
            self.stop()

    def get(self):
        for i in range(self.iterations):
            while len(self.frame_pool) > 20:
                time.sleep(0.1)
            if self.stopped or not self.status: break
            self.load_a_frame()

    def stop(self):
        self.stopped = True


# 模式参数

# debug_mode ： 是否启用debug模式。debug模式下，将会打印详细的日志，并且将启用可视化的渲染。
# 程序运行速度可能会因此减慢，但可以通过日志排查参数设置问题，或者为视频找到更优的参数设置。


#debug_mode = False
def test(video_name,
         mask_name,
         cfg,
         debug_mode,
         work_mode="frontend",
         time_range=(None, None)):
    # load config from cfg json.
    resize_param = cfg["resize_param"]
    visual_param = cfg["visual_param"]
    window_size_s = cfg["window_size_s"]

    detect_cfg = cfg["detect_cfg"]
    meteor_cfg_inp = cfg["meteor_cfg_inp"]

    # load video
    video, mask = load_video_mask(video_name, mask_name, resize_param)
    if (video is None) or (not video.isOpened()):
        raise FileNotFoundError(
            "The file \"%s\" cannot be opened as a supported video format." %
            video_name)

    total_frame, fps = int(video.get(cv2.CAP_PROP_FRAME_COUNT)), video.get(
        cv2.CAP_PROP_FPS)

    # 根据指定头尾跳转指针与结束帧
    start_frame, end_frame = 0, total_frame

    if time_range[0] != None:
        start_frame = max(0, int(time_range[0] / 1000 * fps))
    if time_range[1] != None:
        end_frame = min(int(time_range[1] / 1000 * fps), total_frame)
    if not 0 <= start_frame < end_frame:
        raise ValueError("Invalid start time or end time.")

    set_out_pipe(work_mode)

    # 起点跳转到指定帧
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    #video_loader=video_loader_generator(video,)
    #frame = video_loader.__next__()
    window_size = int(fps * window_size_s)
    # 原则上窗口长度为奇数
    if window_size // 2 == 0:
        window_size += 1
    if detect_cfg["median_sampling_num"] == -1:
        detect_cfg.update(median_skipping=1)
    else:
        assert detect_cfg["median_sampling_num"] >= 3, "You must set median_sampling_num to 3 or larger."
        detect_cfg.update(
            median_skipping=(window_size - 1) // (
                detect_cfg["median_sampling_num"] - 1))
    meteor_cfg = dict(
        min_len=meteor_cfg_inp["min_len"],
        max_interval=meteor_cfg_inp["max_interval"] * fps,
        det_thre=0.5,
        time_range=(meteor_cfg_inp["time_range"][0] * fps,
                    meteor_cfg_inp["time_range"][1] * fps),
        speed_range=meteor_cfg_inp["speed_range"],
        thre2=meteor_cfg_inp["thre2"])

    progout("Total frames = %d ; FPS = %.2f" % (end_frame - start_frame, fps))

    video_reader = VideoRead(
        video,
        iterations=end_frame - start_frame,
        mask=mask,
        resize_param=resize_param)

    window_stack = []

    #t = threading.Thread(
    #    target=video_reader,
    #    name="VideoReader",
    #    args=(video, end_frame - start_frame + 1, mask, resize_param))
    #t.start()

    # 初始化流星收集器
    main_mc = MeteorCollector(**meteor_cfg, fps=fps)

    if work_mode == 'frontend':
        main_iterator = tqdm.tqdm(range(start_frame, end_frame), ncols=50)
    elif work_mode == 'backend':
        main_iterator = range(start_frame, end_frame)

    try:
        video_reader.start()
        for i in main_iterator:
            if work_mode == 'backend' and i % int(fps) == 0:
                progout("Processing: %d" % (i / fps * 1000))
            #status, frame = video.read()
            #if not status:
            #    break
            #frame = preprocessing(frame, mask=mask, resize_param=resize_param)

            if video_reader.stopped and len(video_reader.frame_pool) == 0:
                break

            while (not video_reader.stopped) and len(
                    video_reader.frame_pool) == 0:
                time.sleep(0.1)

            frame = video_reader.frame_pool.pop(0)

            if len(window_stack) == 0:
                window_stack = np.repeat(frame[None, ...], window_size, axis=0)
                #sort_stack = np.argsort(window_stack, axis=0)

            # 栈更新
            window_stack = np.concatenate(
                (window_stack[1:], frame[None, ...]), axis=0)
            #sort_stack = np.concatenate((sort_stack[1:], sort_stack[:1]), axis=0)
            ## Reshape为二维
            #window_stack = np.reshape(window_stack, (L, H * W))
            #sort_stack = np.reshape(sort_stack, (L, H * W))
            ## numba加速的双向冒泡
            #window_stack, sort_stack = sanaas(window_stack, sort_stack, boolmap, L,
            #                                  H, W)
            ## 计算max-median
            #diff_img = np.reshape(window_stack[sort_stack[-1],
            #                                   np.where(sort_stack[-1] >= 0)[0]],
            #                      (H, W))
            ## 形状还原
            #window_stack = np.reshape(window_stack, (L, H, W))
            #sort_stack = np.reshape(sort_stack, (L, H, W))

            # update and calculate
            #window_stack.append(frame)
            #if len(window_stack) > window_size:
            #    window_stack.pop(0)
            #diff_img = np.max(window_stack, axis=0) - np.median(window_stack, axis=0)

            sort_stack = np.sort(
                window_stack[::detect_cfg["median_skipping"]], axis=0)
            diff_img = sort_stack[-1] - sort_stack[len(sort_stack) // 2]

            flag, lines, draw_img = detect_within_window(
                diff_img,
                detect_cfg,
                window_stack[min(len(window_stack) - 1, window_size // 2 + 1)],
                mask,
                visual_param,
                debug_mode=debug_mode)
            if flag:
                output_meteors(main_mc.update(i, lines=lines))
            if debug_mode:
                if (cv2.waitKey(1) & 0xff == ord("q")):
                    break
                draw_img = main_mc.draw_on_img(
                    draw_img, resize_param, cv2.rectangle, ref_zp=visual_param)
                cv2.imshow("DEBUG MODE", draw_img)
    finally:
        video_reader.stop()
        output_meteors(main_mc.update(np.inf, []))
        video.release()
        cv2.destroyAllWindows()
        progout('Video EOF detected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Meteor Detector V1.2')

    parser.add_argument('target', help="input H264 video.")
    parser.add_argument(
        '--cfg', '-C', help="Config file.", default="./config.json")
    parser.add_argument('--mask', '-M', help="Mask image.", default=None)
    parser.add_argument(
        '--start-time',
        help="The start time (ms) of the video.",
        type=int,
        default=None)
    parser.add_argument(
        '--end-time',
        help="The end time (ms) of the video.",
        type=int,
        default=None)
    parser.add_argument(
        '--mode',
        choices=['backend', 'frontend'],
        default='frontend',
        type=str,
        help='Working mode. Logging will change according to the working mode.'
    )
    parser.add_argument(
        '--debug-mode', '-D', help="Apply Debug Mode.", default=False)

    args = parser.parse_args()

    video_name = args.target
    cfg_filename = args.cfg
    mask_name = args.mask
    debug_mode = args.debug_mode
    work_mode = args.mode
    start_time = args.start_time
    end_time = args.end_time
    with open(cfg_filename, mode='r', encoding='utf-8') as f:
        cfg = json.load(f)
    test(
        video_name,
        mask_name,
        cfg,
        debug_mode,
        work_mode,
        time_range=(start_time, end_time))
