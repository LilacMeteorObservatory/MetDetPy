#import logging
import argparse
import datetime
import json
import sys
import time

import cv2
import numpy as np
import tqdm
#import asyncio
import threading

## baseline:
## 42 fps; tp 4/4 ; tn 0/6 ; fp 0/8.

## spring-beta: 11 fps(debug mode);tp 4/4 (some not believeable); tn 3/6 (2 of which are REALLY-HARD!); fp 4/8.(Why???)
## spring-v1: 9 fps (debug mode)/ 16 fps (speed mode); tp 4/4; tn 4/6; fp 8/11.
## spring-v2: 20 fps (no-skipping); 25 fps(median-skipping, fp 6/8)

pi = 3.141592653589793 / 180.0
pt_len_4 = lambda pts: (pts[3] - pts[1])**2 + (pts[2] - pts[0])**2
pt_len_xy = lambda pt1, pt2: (pt1[1] - pt2[1])**2 + (pt1[0] - pt2[0])**2
progout = None

# POSITIVE: 3.85 3.11 3.03 2.68 2.55 2.13 2.61 1.94
# NEGATIVE: 0.49  0.65 2.96 5.08 2.44  1.49 2.69 7.52 19.45 11.18 13.96


class MeteorCollector(object):
    """
    全局的流星统计模块。用于记录和管理所有的响应，整合成正在发生（或已经结束）的检测序列。
    """

    def __init__(self, min_len, max_interval, det_thre, time_range,
                 speed_range, thre2, fps) -> None:
        self.min_len = min_len
        self.max_interval = max_interval
        self.det_thre = det_thre
        self.active_meteor = [
            MeteorSeries(np.inf, [-100, -100, -100, -100], (-np.nan, -np.nan),
                         (-np.nan, -np.nan), np.nan, np.nan)
        ]
        self.waiting_meteor = []
        self.cur_frame = 0
        self.thre2 = thre2
        self.time_range = time_range
        self.speed_range = speed_range
        self.fps = fps

    def update(self, cur_frame, lines):
        """
        更新流星序列的主要函数。

        原则上可以在有新响应时更新，实际为了报告效率，可以无流星时每5-10s执行一次。

        Args:
            cur_frame (_type_): _description_
            lines (_type_): _description_
        """
        # 维护活跃流星序列：将已经超过最长时间检测未响应的潜在流星序列移出，将满足条件的流星放入完成序列。
        self.cur_frame = cur_frame
        temp_waiting_meteor, drop_meteor = [], []
        for ms in self.active_meteor:
            if self.cur_frame - ms.last_activate_frame > self.max_interval:
                if ms.prob_meteor() >= self.det_thre:
                    temp_waiting_meteor.append(ms)
                else:
                    #progout("Dropped:%s" % ms)
                    drop_meteor.append(ms)
                    pass
        # 维护
        for ms in drop_meteor:
            self.active_meteor.remove(ms)
        for ms in temp_waiting_meteor:
            self.active_meteor.remove(ms)
        self.waiting_meteor.extend(temp_waiting_meteor)
        # 整合待导出序列：如果没有活跃的潜在流星，则导出
        if len(self.waiting_meteor) > 0:
            no_prob_met = True
            for ms in self.active_meteor:
                if ms.prob_meteor() >= self.det_thre:
                    no_prob_met = False
                    break
            if no_prob_met:
                for json in self.jsonize_waiting_meteor():
                    progout("Meteor: %s" % json)
                self.waiting_meteor.clear()

        # 对新的line进行判断
        num_activate = len(self.active_meteor)
        for line in lines:
            # 如果某一序列已经开始，则可能是其中间的一部分。
            # 考虑到基本不存在多个流星交接的情况，如果属于某一个，则直接归入即可。
            is_in_series = False
            for ms in self.active_meteor[:num_activate]:
                is_in = ms.may_in_series(line)
                if is_in:
                    ms.update(self.cur_frame, line)
                    is_in_series = True
                    break
            # 如果不属于已存在的序列，并且长度满足触发阈值，则为其构建新的序列开头
            if is_in_series or pt_len_4(line) < self.min_len:
                continue
            #progout("pt %s is not in existing series. Generate new one.." % line)
            self.active_meteor.insert(
                len(self.active_meteor) - 1,
                MeteorSeries(
                    self.cur_frame,
                    line,
                    time_range=self.time_range,
                    speed_range=self.speed_range,
                    max_acceptable_dist=self.thre2,
                    fps=self.fps))

    def jsonize_waiting_meteor(self):
        def init_output_dict(ms, ms_json):
            return dict(
                start_time=ms_json['start_time'],
                end_time=ms_json['end_time'],
                end_frame=ms.end_frame,
                target=[ms_json])

        output_dict = dict()
        final_list = []
        for ms in self.waiting_meteor:
            ms_json = ms.property_json
            if len(output_dict) == 0:
                output_dict = init_output_dict(ms, ms_json)
                continue
            if ms.start_frame < output_dict['end_frame'] + self.max_interval:
                output_dict.update(
                    end_time=ms_json['end_time'], end_frame=ms.end_frame)
                output_dict["target"].append(ms_json)
            else:
                final_list.append(output_dict)
                output_dict = init_output_dict(ms, ms_json)
        if len(output_dict) != 0:
            final_list.append(output_dict)
        final_list = [json.dumps(x) for x in final_list]
        return final_list

    def draw_on_img(self, img, resize_param, ref_zp=(0, 0)):
        #raise NotImplementedError("Global vars are not solved until next update.")
        rescale_param = [y / x for (x, y) in zip(resize_param, ref_zp)]
        for ms in self.active_meteor:
            pt1, pt2 = ms.range
            pt1 = tuple(
                int(x * p + d) for x, p, d in zip(pt1, rescale_param, ref_zp))
            pt2 = tuple(
                int(x * p + d) for x, p, d in zip(pt2, rescale_param, ref_zp))
            img = cv2.rectangle(img, pt1, pt2, [0, 0, 255], 2)
        return img


class MeteorSeries(object):
    """用于整合检测结果，排异和给出置信度的流星序列。

    Args:
        object (_type_): _description_
    """

    def __init__(self, frame, init_box, time_range, speed_range,
                 max_acceptable_dist, fps):
        self.coord_list = []
        self.len_list = []
        self.coord_list.append(self.box2coord(init_box))
        self.len_list.append(pt_len_4(init_box))
        self.start_frame = frame
        self.end_frame = frame
        self.last_activate_frame = frame
        self.max_acceptable_dist = max_acceptable_dist
        self.time_range = time_range
        self.speed_range = speed_range
        self.fps = fps

    def __repr__(self) -> str:
        return "Duration %s frames; (Dist=%s); speed=%.2f px(s)/frame; \"%s - %s : %s - %s\"" % (
            self.duration, self.dist, self.speed, self.start_frame,
            self.last_activate_frame, self.range[0], self.range[1])

    @property
    def property_json(self) -> dict:
        return dict(
            start_time=self.frame2ts(self.start_frame),
            end_time=self.frame2ts(self.end_frame),
            last_activate_time=self.frame2ts(self.last_activate_frame),
            duration=self.duration,
            speed=self.speed,
            dist=self.dist,
            pt1=self.range[0],
            pt2=self.range[1])

    @property
    def duration(self):
        return self.last_activate_frame - self.start_frame + 1

    @property
    def range(self):
        return [
            int(min([x[0] for x in self.coord_list])),
            int(min([x[1] for x in self.coord_list]))
        ], [
            int(max([x[0] for x in self.coord_list])),
            int(max([x[1] for x in self.coord_list]))
        ]

    @property
    def dist(self):
        return pt_len_xy(*self.range)**(1 / 2)

    @property
    def speed(self):
        return self.dist / (self.end_frame - self.start_frame)

    def frame2ts(self, frame):
        return datetime.datetime.strftime(
            datetime.datetime.utcfromtimestamp(frame / self.fps),
            "%H:%M:%S.%f")

    def box2coord(cls, box):
        return [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]

    def update(self, new_frame, new_box):
        pt1, pt2 = new_box[:2], new_box[2:]
        (x1, y1), (x2, y2) = self.range
        if not (((x1 <= pt1[0] <= x2) and (y1 <= pt1[1] <= y2)) and
                ((x1 <= pt2[0] <= x2) and (y1 <= pt2[1] <= y2))):
            self.end_frame = new_frame
        self.last_activate_frame = new_frame
        self.coord_list.extend([pt1, pt2])
        self.len_list.append(pt_len_4(new_box))

    def may_in_series(self, new_box):
        # 策略一：最后近邻法（对于有尾迹的判断不准确）
        #if pt_len(self.box2coord(new_box)+self.coord_list[-1])<self.max_acceptable_dist:
        #    return True
        # 策略二：近邻法（对于距离中间点近的，采取收入但不作为边界点策略）
        for in_pt in self.coord_list:
            if pt_len_xy(self.box2coord(new_box),
                         in_pt) < self.max_acceptable_dist:
                return True
        return False

    def is_in_range(self, value, range_tuple):
        if range_tuple[0] <= value <= range_tuple[1]:
            return True
        return False

    def prob_meteor(self):
        # 自身为流星序列的概率。
        # 拟借助几个指标
        # 1. 总速度/总长度
        # 2. 平均响应长度（暂未实现）
        # 3. 直线拟合情况（暂未实现）

        # 排除总时长过长/过短
        if not self.is_in_range(self.duration, self.time_range):
            return 0
        # 排除速度过快/过慢
        if not self.is_in_range(self.speed, self.speed_range):
            return 0

        return 1


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

            if video_reader.stopped and len(video_reader.frame_pool)==0:
                break
            
            while (not video_reader.stopped) and len(video_reader.frame_pool)==0:
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

            sort_stack = np.sort(window_stack, axis=0)
            diff_img = sort_stack[-1] - sort_stack[window_size // 2]

            flag, lines, draw_img = detect_within_window(
                diff_img,
                detect_cfg,
                window_stack[min(len(window_stack) - 1, window_size // 2 + 1)],
                mask,
                visual_param,
                debug_mode=debug_mode)
            if flag:
                main_mc.update(i, lines=lines)
            if debug_mode:
                if (cv2.waitKey(1) & 0xff == ord("q")):
                    break
                draw_img = main_mc.draw_on_img(
                    draw_img, resize_param, ref_zp=visual_param)
                cv2.imshow("DEBUG MODE", draw_img)
    finally:
        video_reader.stop()
        main_mc.update(np.inf, [])
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
        '-W',
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
