#import logging
import argparse
import asyncio
import json
import time
from functools import partial
from math import floor, trunc

import cv2
import numpy as np
import tqdm
from munch import Munch

from MetLib import init_detector, init_stacker
from MetLib.MeteorLib import MeteorCollector
from MetLib.utils import (init_exp_time, load_video_and_mask, preprocessing,
                          set_out_pipe)
from MetLib.VideoLoader import ThreadVideoReader

## baseline:
## 42 fps; tp 4/4 ; tn 0/6 ; fp 0/8.

## spring-beta: 11 fps(debug mode);tp 4/4 (some not believeable); tn 3/6 (2 of which are REALLY-HARD!); fp 4/8.(Why???)
## spring-v1: 9 fps (debug mode)/ 16 fps (speed mode); tp 4/4; tn 4/6; fp 8/11.
## spring-v2: 20 fps (no-skipping); 25 fps(median-skipping, fp 6/8)

# POSITIVE: 3.85 3.11 3.03 2.68 2.55 2.13 2.61 1.94
# NEGATIVE: 0.49  0.65 2.96 5.08 2.44  1.49 2.69 7.52 19.45 11.18 13.96


def output_meteors(update_info, stream):
    met_lst, drop_lst = update_info
    for met in met_lst:
        stream("Meteor: %s" % met)
    for met in drop_lst:
        stream("Dropped: %s" % met)


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

# 模式参数

# debug_mode ： 是否启用debug模式。debug模式下，将会打印详细的日志，并且将启用可视化的渲染。
# 程序运行速度可能会因此减慢，但可以通过日志排查参数设置问题，或者为视频找到更优的参数设置。

'''


async def detect_video(video_name,
                       mask_name,
                       cfg,
                       debug_mode,
                       work_mode="frontend",
                       time_range=(None, None)):
    # load config from cfg json.
    resize_param = cfg.resize_param
    meteor_cfg_inp = cfg["meteor_cfg_inp"]

    # set output mode
    progout = set_out_pipe(work_mode)

    # load video
    video, mask = load_video_and_mask(video_name, mask_name, resize_param)
    if (video is None) or (not video.isOpened()):
        raise FileNotFoundError(
            "The file \"%s\" cannot be opened as a supported video format." %
            video_name)

    # Acquire exposure time and eqirvent FPS(eq_fps)
    total_frame, fps = int(video.get(cv2.CAP_PROP_FRAME_COUNT)), video.get(
        cv2.CAP_PROP_FPS)
    if cfg.stacker == "SimpleStacker":
        progout(
            "Ignore the option \"exp_time\" when appling \"SimpleStacker\".")
        exp_time, exp_frame, eq_fps, eq_int_fps = 1 / fps, 1, fps, int(fps)
    else:
        progout("Parsing \"exp_time\"=%s" % (cfg.exp_time))
        exp_time = init_exp_time(cfg.exp_time, video, mask)
        exp_frame, eq_fps, eq_int_fps = trunc(
            exp_time * fps), 1 / exp_time, floor(1 / exp_time)
    progout("Apply exposure time of %.2fs." % (exp_time))
    # 根据指定头尾跳转指针与结束帧
    start_frame, end_frame = 0, total_frame
    start_time, end_time = time_range
    if start_time != None:
        start_frame = max(0, int(start_time / 1000 * fps))
    if end_time != None:
        end_frame = min(int(end_time / 1000 * fps), total_frame)
    if not 0 <= start_frame < end_frame:
        raise ValueError("Invalid start time or end time.")
    # 起点跳转到指定帧
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    progout("Total frames = %d ; FPS = %.2f (rFPS = %.2f)" %
            (end_frame - start_frame, fps, eq_fps))

    # Init stacker_manager
    stack_manager = init_stacker(cfg.stacker, cfg.stacker_cfg, exp_frame)

    # Init detector
    detector = init_detector(cfg.detector, cfg.detect_cfg, debug_mode, eq_fps)

    # Init meteor collector
    # TODO: To be renewed
    meteor_cfg_inp = Munch(meteor_cfg_inp)
    meteor_cfg = dict(
        min_len=meteor_cfg_inp.min_len,
        max_interval=meteor_cfg_inp.max_interval * fps,
        det_thre=0.5,
        time_range=(meteor_cfg_inp.time_range[0] * fps,
                    meteor_cfg_inp.time_range[1] * fps),
        speed_range=meteor_cfg_inp.speed_range,
        thre2=meteor_cfg_inp.thre2)
    main_mc = MeteorCollector(**meteor_cfg, fps=fps)

    # Init videoReader
    video_reader = ThreadVideoReader(
        video,
        iterations=end_frame - start_frame,
        pre_func=partial(preprocessing, mask=mask, resize_param=resize_param))

    # Init main iterator
    main_iterator = range(start_frame, end_frame, exp_frame)
    if work_mode == 'frontend':
        main_iterator = tqdm.tqdm(main_iterator, ncols=100)

    try:
        t0 = time.time()
        video_reader.start()
        for i in main_iterator:
            # Logging for backend only.
            # TODO: Use Logging module to replace progout
            if work_mode == 'backend' and i % eq_int_fps == 0:
                progout("Processing: %d" % (int(1000 * i / fps)))

            #print(len(video_reader.frame_pool))

            if video_reader.stopped and len(video_reader.frame_pool) == 0:
                break

            # TODO: Replace with API of video_reader.
            stack_manager.update(video_reader, detector)

            #TODO: Mask, visual
            flag, lines = detector.detect()

            if flag:
                output_meteors(main_mc.update(i, lines=lines), progout)
            if debug_mode:
                if (cv2.waitKey(1) & 0xff == ord("q")):
                    break
                draw_img = main_mc.draw_on_img(
                    stack_manager.cur_frame)
                cv2.imshow("DEBUG MODE", draw_img)

    finally:
        video_reader.stop()
        output_meteors(main_mc.update(np.inf, []), progout)
        video.release()
        cv2.destroyAllWindows()
        progout('Video EOF detected.')
        progout("Time cost: %.4ss." % (time.time() - t0))


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
        '--debug',
        '-D',
        action='store_true',
        help="Apply Debug Mode",
        default=False)

    args = parser.parse_args()

    video_name = args.target
    cfg_filename = args.cfg
    mask_name = args.mask
    debug_mode = args.debug
    work_mode = args.mode
    start_time = args.start_time
    end_time = args.end_time
    with open(cfg_filename, mode='r', encoding='utf-8') as f:
        cfg = Munch(json.load(f))
    # async main loop
    loop = asyncio.get_event_loop()
    tasks = [
        detect_video(
            video_name,
            mask_name,
            cfg,
            debug_mode,
            work_mode,
            time_range=(start_time, end_time))
    ]
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()
