#import logging
import argparse
import json
import time
from typing import Any

import tqdm
from easydict import EasyDict

from MetLib import get_loader, get_warpper, get_detector
from MetLib.MeteorLib import MeteorCollector
from MetLib.MetLog import get_default_logger, set_default_logger
from MetLib.utils import frame2time, output_meteors, VERSION
from MetLib.MetVisu import OpenCVMetVisu
from MetLib.Detector import LineDetector

def detect_video(video_name,
                 mask_name,
                 cfg,
                 debug_mode,
                 work_mode="frontend",
                 time_range=(None, None)):

    # set output mode
    set_default_logger(debug_mode, work_mode)
    logger = get_default_logger()
    logger.start()

    # initialization
    try:
        t0 = time.time()

        # parse preprocessing params
        VideoLoaderCls = get_loader(cfg.loader.name)
        VideoWarpperCls = get_warpper(cfg.loader.warpper)
        DetectorCls = get_detector(cfg.detector.name)
        resize_option = cfg.loader.resize
        exp_option = cfg.loader.exp_time
        merge_func = cfg.loader.merge_func
        grayscale = cfg.loader.grayscale
        start_time, end_time = time_range
        if issubclass(DetectorCls, LineDetector):
            assert grayscale, "Require grayscale ON when using subclass of LineDetector."
        # Init VideoLoader
        # Since v2.0.0, VideoLoader will control most video-related varibles and functions.
        video_loader = VideoLoaderCls(VideoWarpperCls,
                                    video_name,
                                    mask_name,
                                    resize_option,
                                    start_time=start_time,
                                    end_time=end_time,
                                    grayscale=grayscale,
                                    exp_option=exp_option,
                                    merge_func=merge_func)
        logger.info(video_loader.summary())

        # get properties from VideoLoader
        start_frame, end_frame = video_loader.start_frame, video_loader.end_frame
        fps, exp_frame, eq_fps, eq_int_fps, exp_time = (
            video_loader.fps, video_loader.exp_frame, video_loader.eq_fps,
            video_loader.eq_int_fps, video_loader.exp_time)

        logger.info(
            f"Preprocessing finished. Time cost: {(time.time() - t0):.1f}s.")
        # wait for logger clear
        while not logger.is_empty:
            continue

        # Init detector
        # TODO: 优化写法
        cfg_det = cfg.detector
        if cfg_det.bi_cfg.sensitivity == "high":
            cfg_det.hough_cfg.max_gap = 10
        cfg_det.img_mask = video_loader.mask
        cfg_det.fps = eq_fps
        detector = DetectorCls(window_sec=cfg_det.window_sec,
                                fps=cfg_det.fps,
                                mask=cfg_det.img_mask,
                                bi_cfg=cfg_det.bi_cfg,
                                hough_cfg=cfg_det.hough_cfg,
                                dynamic_cfg=cfg_det.dynamic_cfg)

        # Init meteor collector
        # TODO: To be renewed
        meteor_cfg = cfg.meteor_cfg
        # 修改属性
        meteor_cfg.max_interval *= fps
        meteor_cfg.time_range[0] *= fps
        meteor_cfg.time_range[1] *= fps
        meteor_cfg.thre2 *= exp_frame

        main_mc = MeteorCollector(**meteor_cfg,
                                  eframe=exp_frame,
                                  fps=fps,
                                  runtime_size=video_loader.runtime_size,
                                  raw_size=video_loader.raw_size)
        # Init visualizer
        # TODO: 参数暂未完全支持参数化设置。
        visual_manager = OpenCVMetVisu(exp_time=exp_time,
                                       resolution=video_loader.runtime_size,
                                       flag=debug_mode)
        # Init main iterator
        main_iterator = range(start_frame, end_frame, exp_frame)
        if work_mode == 'frontend':
            main_iterator = tqdm.tqdm(main_iterator, ncols=100)
    except Exception as e:
        logger.error(
            'Fatal error occured when initializing. MetDetPy will exit.')
        logger.stop()
        raise e
    # MAIN DETECTION PART
    t1 = time.time()
    try:
        video_loader.start()
        for i in main_iterator:
            # Logging for backend only.
            if work_mode == 'backend' and (
                (i - start_frame) // exp_frame) % eq_int_fps == 0:
                logger.processing(frame2time(i, fps))

            x = video_loader.pop()

            if (video_loader.stopped or x is None):
                break

            detector.update(x)
            #TODO: Mask, visual
            lines, detect_info = detector.detect()

            if len(lines) or (((i - start_frame) // exp_frame) % eq_int_fps
                              == 0):
                met_info = main_mc.update(i, lines=lines)
                output_meteors(met_info)

            detect_info["info"] += main_mc.draw_on_img(frame_num=i)
            
            visual_manager.display_a_frame(detect_info)
            if visual_manager.manual_stop:
                logger.info('Manual interrupt signal detected.')
                break
        # 仅正常结束时（即 手动结束或视频读取完）打印。
        if not visual_manager.manual_stop:
            # TODO: 改下描述
            logger.info('VideoLoader-stop detected.')
    except Exception as e:
        print(e)
        raise e
    finally:
        video_loader.release()
        output_meteors(main_mc.clear())
        visual_manager.stop()
        logger.info("Time cost: %.4ss." % (time.time() - t1))
        logger.stop()

    return main_mc.ended_meteor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f'MetDetPy{VERSION}')
    # TODO: Add More Details.
    parser.add_argument('target', help="input video. Support H264, HEVC, etc.")
    parser.add_argument('--cfg',
                        '-C',
                        help="Config file.",
                        default="./config.json")
    parser.add_argument('--mask', '-M', help="Mask image.", default=None)

    parser.add_argument('--start-time',
                        help="The start time (ms) of the video.",
                        type=str,
                        default=None)
    parser.add_argument('--end-time',
                        help="The end time (ms) of the video.",
                        type=str,
                        default=None)
    parser.add_argument(
        '--mode',
        choices=['backend', 'frontend'],
        default='frontend',
        type=str,
        help='Working mode. Logging will change according to the working mode.'
    )
    parser.add_argument('--debug',
                        '-D',
                        action='store_true',
                        help="Apply Debug Mode",
                        default=False)

    parser.add_argument('--resize',
                        help="Running-time resolution",
                        type=str,
                        default=None)
    parser.add_argument(
        '--exp-time',
        help=
        "The exposure time (s) of the video. \"auto\", \"real-time\",\"slow\" are also supported.",
        type=str,
        default=None)
    parser.add_argument('--adaptive-thre',
                        default=None,
                        type=str,
                        help="Apply adaptive binary threshold.")

    group_bi = parser.add_mutually_exclusive_group(required=False)
    group_bi.add_argument('--bi-thre',
                          type=int,
                          default=None,
                          help="Constant binary threshold value.")

    group_bi.add_argument('--sensitivity',
                          type=str,
                          default=None,
                          help="The sensitivity of detection.")

    args = parser.parse_args()

    video_name = args.target
    cfg_filename = args.cfg
    mask_name = args.mask
    debug_mode = args.debug
    sensitivity = args.sensitivity
    bi_thre = args.bi_thre
    adaptive = args.adaptive_thre
    work_mode = args.mode
    start_time = args.start_time
    end_time = args.end_time
    exp_time = args.exp_time
    resize = args.resize
    with open(cfg_filename, mode='r', encoding='utf-8') as f:
        cfg: Any = EasyDict(json.load(f))

    # 当通过参数的指定部分选项时，替代配置文件中的缺省项
    # replace config value
    if exp_time:
        cfg.loader.exp_time = exp_time
    if resize:
        cfg.loader.resize = resize
    if adaptive:
        assert adaptive in ["on", "off"
                            ], "adaptive_thre should be set \"on\" or \"off\"."
        cfg.detector.bi_cfg.adaptive_bi_thre = {"on": True, "off": False}[adaptive]
    if sensitivity:
        cfg.detector.bi_cfg.sensitivity = sensitivity
    if bi_thre:
        cfg.detector.bi_cfg.init_value = bi_thre

    # Preprocess start_time and end_time to int

    detect_video(video_name,
                 mask_name,
                 cfg,
                 debug_mode,
                 work_mode,
                 time_range=(start_time, end_time))