#import logging
import argparse
import json
import time
from typing import Any

import cv2
import tqdm
from easydict import EasyDict

from MetLib import get_loader, get_warpper, get_detector
from MetLib.MeteorLib import MeteorCollector
from MetLib.MetLog import get_default_logger, set_default_logger
from MetLib.utils import frame2time, output_meteors, VERSION


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

    try:
        t0 = time.time()

        # parse preprocessing params
        video_loader = get_loader(cfg.loader.name)
        video_warpper = get_warpper(cfg.loader.warpper)
        resize_option = cfg.loader.resize
        exp_option = cfg.loader.exp_time
        merge_func = cfg.loader.merge_func
        start_time, end_time = time_range

        # Init VideoLoader
        # Since v2.0.0, VideoLoader will control most video-related varibles and functions.
        video_reader = video_loader(video_warpper,
                                    video_name,
                                    mask_name,
                                    resize_option,
                                    start_time=start_time,
                                    end_time=end_time,
                                    grayscale=True,
                                    exp_option=exp_option,
                                    merge_func=merge_func)
        logger.info(video_reader.summary())

        # get properties of VideoLoader
        start_frame, end_frame = video_reader.start_frame, video_reader.end_frame
        fps, exp_frame, eq_fps, eq_int_fps, exp_time = (
            video_reader.fps, video_reader.exp_frame, video_reader.eq_fps,
            video_reader.eq_int_fps, video_reader.exp_time)

        # wait for logger clear
        while not logger.is_empty:
            continue

        logger.info(
            f"Preprocessing finished. Time cost: {(time.time() - t0):.1f}s.")

        # Init detector
        if cfg.detect_cfg.bi_cfg.sensitivity == "high":
            cfg.detector.hough_cfg.max_gap = 10
        cfg.detector.img_mask = video_reader.mask
        cfg.detector.fps = eq_fps
        detector = get_detector(cfg.detector)

        # Init meteor collector
        # TODO: To be renewed
        meteor_cfg = cfg.meteor_cfg
        # 修改属性
        meteor_cfg.max_interval *= fps
        meteor_cfg.time_range[0] *= fps
        meteor_cfg.time_range[1] *= fps
        meteor_cfg.thre2 *= exp_frame

        # TODO: alias, which is not elegant.
        # To be removed in the near future.
        if meteor_cfg.get("pos_threshold", None):
            meteor_cfg.det_thre = meteor_cfg.pos_threshold
            del meteor_cfg["pos_threshold"]

        main_mc = MeteorCollector(**meteor_cfg,
                                  eframe=exp_frame,
                                  fps=fps,
                                  runtime_size=video_reader.runtime_size,
                                  raw_size=video_reader.raw_size)

        # Init main iterator
        main_iterator = range(start_frame, end_frame, exp_frame)
        if work_mode == 'frontend':
            main_iterator = tqdm.tqdm(main_iterator, ncols=100)
    except Exception as e:
        logger.error(
            'Fatal error occured when initializing. MetDetPy will exit.')
        logger.stop()
        raise e
    try:
        t0 = time.time()
        video_reader.start()
        for i in main_iterator:
            # Logging for backend only.
            if work_mode == 'backend' and (
                (i - start_frame) // exp_frame) % eq_int_fps == 0:
                logger.processing(frame2time(i, fps))
            if video_reader.stopped:
                break

            detector.update(video_reader.pop())

            #TODO: Mask, visual
            lines, img_visu = detector.detect()

            if len(lines) or (((i - start_frame) // exp_frame) % eq_int_fps
                              == 0):
                output_meteors(main_mc.update(i, lines=lines))
            if debug_mode:
                if (cv2.waitKey(int(exp_time * 400)) & 0xff == ord("q")):
                    logger.info('Keyboard interrupt detected.')
                    break
                # img_visu is used to get the image,
                # and is only executed when visualizing.
                draw_img = main_mc.draw_on_img(img_visu(), frame_num=i)
                #cv2.imwrite("test/frame_%s.jpg"%i,draw_img)
                cv2.imshow("Debug Window (Press Q to exit)", draw_img)
        logger.info('Video EOF detected.')
    finally:
        video_reader.stop()
        video_reader.release()
        output_meteors(main_mc.clear())
        cv2.destroyAllWindows()
        logger.info("Time cost: %.4ss." % (time.time() - t0))
        logger.stop()

    return main_mc.ended_meteor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f'Meteor Detector {VERSION}')
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
        cfg.detect_cfg.adaptive_bi_thre = {"on": True, "off": False}[adaptive]
    if sensitivity:
        cfg.detect_cfg.bi_cfg.sensitivity = sensitivity
    if bi_thre:
        cfg.detect_cfg.bi_cfg.init_value = bi_thre

    # Preprocess start_time and end_time to int

    detect_video(video_name,
                 mask_name,
                 cfg,
                 debug_mode,
                 work_mode,
                 time_range=(start_time, end_time))