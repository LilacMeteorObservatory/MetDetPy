#import logging
import argparse
import json
import time
from functools import partial
from math import floor

import cv2
import tqdm

from MetLib.Detector import init_detector
from MetLib.MeteorLib import MeteorCollector
from MetLib.utils import (Munch, init_exp_time, load_video_and_mask,
                          preprocessing)
from MetLib.VideoLoader import ThreadVideoReader
from MetLib.MetLog import get_default_logger, set_default_logger


def output_meteors(update_info):
    logger = get_default_logger()
    met_lst, drop_lst = update_info
    for met in met_lst:
        logger.meteor(met)
    for met in drop_lst:
        logger.dropped(met)


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

        # deprecated warning
        if getattr(cfg, "exp_time", None) or getattr(cfg, "resize_param",
                                                     None):
            logger.warning(
                """\"exp_time\" and \"resize_param\" will be moved to \"preprocessing\" in the future.
            Config \"exp_time\", \"resize_param\" and \"merge_func\" in \"preprocessing\" instead."""
            )

        if getattr(cfg, "stacker_cfg", None) or getattr(cfg, "stacker", None):
            logger.warning(
                """\"stacker\" and \"stacker_cfg\" will be deprecated in the future.
            Use \"exp_time\"=\"auto\" and \"merge_func\" instead.""")

        # parse preprocessing params
        cfg.preprocessing = Munch(cfg.preprocessing)
        resize_option = cfg.preprocessing.resize_param
        exp_option = cfg.preprocessing.exp_time
        merge_func = cfg.preprocessing.merge_func

        # load video, init video_reader
        video, mask = load_video_and_mask(video_name, mask_name, resize_option)
        resize_param = list(reversed(mask.shape))
        logger.info(
            f"Raw resolution = {video.size}; apply running-time resolution = {resize_param}."
        )

        # get accurate start_frame and end_frame according to the input arguments.
        total_frame, fps = video.num_frames, video.fps
        start_frame, end_frame = 0, video.num_frames
        start_time, end_time = time_range
        if start_time != None:
            start_frame = max(0, int(start_time / 1000 * fps))
        if end_time != None:
            end_frame = min(int(end_time / 1000 * fps), total_frame)
        if not 0 <= start_frame < end_frame:
            raise ValueError("Invalid start time or end time.")

        # Init videoReader
        video_reader = ThreadVideoReader(video,
                                         start_frame=start_frame,
                                         iterations=end_frame - start_frame,
                                         pre_func=partial(
                                             preprocessing,
                                             mask=mask,
                                             resize_param=resize_param),
                                         exp_frame=1,
                                         merge_func=merge_func)

        # Acquire exposure time and eqirvent FPS(eq_fps)
        # SimpleStacker is left only for compatibility.
        if getattr(cfg, "stacker", None) == "SimpleStacker":
            logger.warning(
                "Ignore the option \"exp_time\" when appling \"SimpleStacker\"."
            )
            exp_time, exp_frame, eq_fps, eq_int_fps = 1 / fps, 1, fps, int(fps)
        else:
            logger.info("Parsing \"exp_time\"=%s" % (exp_option))
            exp_time = init_exp_time(exp_option,
                                     video_reader,
                                     upper_bound=0.5)
            exp_frame, eq_fps, eq_int_fps = int(round(
                exp_time * fps)), 1 / exp_time, floor(1 / exp_time)
            
        min_time_flag = 1000 * exp_frame * eq_int_fps / fps
        logger.info(
            f"Apply exposure time of {exp_time:.2f}s. (MinTimeFlag = {min_time_flag})"
        )
        logger.info("Total frames = %d ; FPS = %.2f (rFPS = %.2f)" %
                    (end_frame - start_frame, fps, eq_fps))
        logger.info(
            f"Preprocessing finished. Time cost: {(time.time() - t0):.1f}s.")
        # Reset video reader for main progress.
        video_reader.reset(start_frame=start_frame,
                           iterations=end_frame - start_frame,
                           exp_frame=exp_frame)

        # Init detector
        if cfg.detect_cfg["bi_cfg"]["sensitivity"]=="high":
            cfg.detect_cfg["max_gap"]=10
        cfg.detect_cfg.update(img_mask=mask)
        detector = init_detector(cfg.detector, cfg.detect_cfg, eq_fps)

        # Init meteor collector
        # TODO: To be renewed
        # TODO: Update My Munch
        meteor_cfg = Munch(cfg.meteor_cfg)
        meteor_cfg = dict(min_len=meteor_cfg.min_len,
                          max_interval=meteor_cfg.max_interval * fps,
                          det_thre=0.5,
                          time_range=[
                              meteor_cfg.time_range[0] * fps,
                              meteor_cfg.time_range[1] * fps
                          ],
                          speed_range=meteor_cfg.speed_range,
                          drct_range=meteor_cfg.drct_range,
                          thre2=meteor_cfg.thre2 * exp_frame)
        main_mc = MeteorCollector(**meteor_cfg,
                                  eframe=exp_frame,
                                  fps=fps,
                                  runtime_size=resize_param,
                                  raw_size=video.size)

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
                logger.processing(int(1000 * i / fps))

            if video_reader.stopped and video_reader.is_empty:
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
        output_meteors(main_mc.clear())
        video.release()
        cv2.destroyAllWindows()
        logger.info("Time cost: %.4ss." % (time.time() - t0))
        logger.stop()

    return main_mc.ended_meteor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Meteor Detector V1.2.4')

    parser.add_argument('target', help="input H264 video.")
    parser.add_argument('--cfg',
                        '-C',
                        help="Config file.",
                        default="./config.json")
    parser.add_argument('--mask', '-M', help="Mask image.", default=None)

    parser.add_argument('--start-time',
                        help="The start time (ms) of the video.",
                        type=int,
                        default=None)
    parser.add_argument('--end-time',
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
    resize_param = args.resize
    with open(cfg_filename, mode='r', encoding='utf-8') as f:
        cfg = Munch(json.load(f))

    # convert old-style config to new format by adding preprocessing automatically.
    if getattr(cfg, "preprocessing", None) == None:
        cfg.add(
            "preprocessing",
            dict(
                exp_time=cfg.exp_time,
                resize_param=cfg.resize_param,
                merge_func=getattr(cfg, "stacker_cfg",
                                   dict()).get("pfunc", None),
            ))

    # replace config value
    if exp_time:
        cfg.preprocessing["exp_time"] = exp_time
    if resize_param:
        cfg.preprocessing["resize_param"] = resize_param
    if adaptive:
        assert adaptive in ["on", "off"
                            ], "adaptive_thre should be set \"on\" or \"off\"."
        adaptive = {"on": True, "off": False}[adaptive]
        cfg.detect_cfg["adaptive_bi_thre"] = adaptive

    if sensitivity:
        cfg.detect_cfg["bi_cfg"]["sensitivity"] = sensitivity
    if bi_thre:
        cfg.detect_cfg["bi_cfg"]["init_value"] = bi_thre

    detect_video(video_name,
                 mask_name,
                 cfg,
                 debug_mode,
                 work_mode,
                 time_range=(start_time, end_time))
    # async main loop
    #loop = asyncio.get_event_loop()
    #tasks = [
    #    detect_video(
    #        video_name,
    #        mask_name,
    #        cfg,
    #        debug_mode,
    #        work_mode,
    #        time_range=(start_time, end_time))
    #]
    #loop.run_until_complete(asyncio.wait(tasks))
    #loop.close()
