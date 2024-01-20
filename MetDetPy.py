#import logging
import argparse
import json
import time
from typing import Any

import tqdm
from easydict import EasyDict

from MetLib import get_detector, get_loader, get_wrapper
from MetLib.Detector import LineDetector
from MetLib.MeteorLib import MeteorCollector
from MetLib.MetLog import get_default_logger, set_default_logger
from MetLib.MetVisu import OpenCVMetVisu
from MetLib.utils import VERSION, frame2time, relative2abs_path, SWITCH2BOOL


def detect_video(video_name,
                 mask_name,
                 cfg,
                 debug_mode=False,
                 visual_mode=False,
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
        VideoWrapperCls = get_wrapper(cfg.loader.wrapper)
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
        video_loader = VideoLoaderCls(VideoWrapperCls,
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
        cfg_det = cfg.detector
        detector = DetectorCls(window_sec=cfg_det.window_sec,
                               fps=eq_fps,
                               mask=video_loader.mask,
                               cfg=cfg_det.cfg,
                               logger=logger)

        # Init meteor collector
        meteor_cfg = cfg.collector.meteor_cfg
        recheck_cfg = cfg.collector.recheck_cfg
        recheck_loader = None
        if recheck_cfg.switch:
            recheck_loader = VideoLoaderCls(VideoWrapperCls,
                                            video_name,
                                            mask_name,
                                            resize_option,
                                            grayscale=False,
                                            exp_option=exp_time,
                                            merge_func=merge_func)

        meteor_collector = MeteorCollector(meteor_cfg,
                                  eframe=exp_frame,
                                  fps=fps,
                                  runtime_size=video_loader.runtime_size,
                                  raw_size=video_loader.raw_size,
                                  recheck_cfg=recheck_cfg,
                                  video_loader=recheck_loader,
                                  logger=logger)

        # Init visualizer
        # TODO: 可视化模块暂未完全支持参数化设置。
        visual_manager = OpenCVMetVisu(exp_time=exp_time,
                                       resolution=video_loader.runtime_size,
                                       flag=visual_mode)
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
    tot_get_time = 0
    try:
        video_loader.start()
        for i in main_iterator:
            # Logging for backend only.
            if work_mode == 'backend' and (
                (i - start_frame) // exp_frame) % eq_int_fps == 0:
                logger.processing(frame2time(i, fps))
            t0 = time.time()
            x = video_loader.pop()
            tot_get_time += (time.time() - t0)
            if (video_loader.stopped or x is None):
                break

            detector.update(x)
            #TODO: Mask, visual
            lines, cates, detect_info = detector.detect()

            if len(lines) or (((i - start_frame) // exp_frame) % eq_int_fps
                              == 0):
                meteor_collector.update(i, lines=lines, cates=cates)

            detect_info["info"] += meteor_collector.draw_on_img(frame_num=i)

            visual_manager.display_a_frame(detect_info)
            if visual_manager.manual_stop:
                logger.info('Manual interrupt signal detected.')
                break

        # 仅正常结束时（即 手动结束或视频读取完）打印。
        if not visual_manager.manual_stop:
            logger.info('VideoLoader-stop detected.')
    except Exception as e:
        logger.error(e)
        raise e
    finally:
        video_loader.release()
        meteor_collector.clear()
        # TODO: 不是很美观。后续封装一下
        meteor_collector.met_exporter.export_loop.join()
        visual_manager.stop()
        logger.info("Time cost: %.4ss." % (time.time() - t1))
        logger.debug(f"Total Pop Waiting Time = {tot_get_time:.4f}s.")
        logger.stop()

    return meteor_collector.ended_meteor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f'MetDetPy {VERSION}')
    # TODO: Add More Details.
    parser.add_argument(
        'target',
        help="input video. Support common video encoding like H264, HEVC, etc."
    )
    parser.add_argument(
        '--cfg',
        '-C',
        help="Path to the config file.",
        default=relative2abs_path("./config/m3det_normal.json"))
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

    parser.add_argument('--visual',
                        '-V',
                        action='store_true',
                        help="Apply Visual Mode",
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
                        choices=['on', 'off'],
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

    parser.add_argument('--recheck',
                        type=str,
                        choices=['backend', 'frontend'],
                        default=None,
                        help="Apply recheck before the result is printed"
                        "(the model must specified in the config file).")

    parser.add_argument('--save-rechecked-img',
                        type=str,
                        help="Save rechecked images to the given path.")

    args = parser.parse_args()

    with open(args.cfg, mode='r', encoding='utf-8') as f:
        cfg: Any = EasyDict(json.load(f))
    # TODO: 添加对于cfg的格式检查

    # 当通过参数的指定部分选项时，替代配置文件中的缺省项
    # replace cfg value
    if args.exp_time:
        cfg.loader.exp_time = args.exp_time
    if args.resize:
        cfg.loader.resize = args.resize

    # 与二值化有关的参数仅在使用直线型检测器时生效
    if isinstance(get_loader(cfg.loader.name), LineDetector):
        if args.adaptive_thre:
            cfg.detector.bi_cfg.adaptive_bi_thre = SWITCH2BOOL[
                args.adaptive_thre]
        if args.sensitivity:
            cfg.detector.cfg.binary.sensitivity = args.sensitivity
            # TODO: to be changed in the future.
            print("\"sensitivity\" is considered to be rebuilt in v2.0.0."
                  " Avoid use this. Instead, use config files.")
        if args.bi_thre:
            cfg.detector.bi_cfg.init_value = args.bi_thre

    if args.recheck:
        cfg.collector.recheck_cfg.switch = SWITCH2BOOL[args.recheck]
    if args.save_rechecked_img:
        cfg.collector.recheck_cfg.save_path = args.save_rechecked_img

    detect_video(args.target,
                 args.mask,
                 cfg,
                 args.debug,
                 args.visual,
                 work_mode=args.mode,
                 time_range=(args.start_time, args.end_time))