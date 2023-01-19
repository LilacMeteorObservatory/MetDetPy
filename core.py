#import logging
import argparse
import json
import time
from functools import partial
from math import floor

import cv2
import numpy as np
import tqdm

from MetLib import init_detector
from MetLib.MeteorLib import MeteorCollector
from MetLib.utils import (Munch, init_exp_time, load_video_and_mask,
                          preprocessing, set_out_pipe)
from MetLib.VideoLoader import ThreadVideoReader

## baseline:
## 42 fps; tp 4/4 ; tn 0/6 ; fp 0/8.

## spring-beta: 11 fps(debug mode);tp 4/4 (some not believeable); tn 3/6 (2 of which are REALLY-HARD!); fp 4/8.(Why???)
## spring-v1: 9 fps (debug mode)/ 16 fps (speed mode); tp 4/4; tn 4/6; fp 8/11.
## spring-v2: 20 fps (no-skipping); 25 fps(median-skipping, fp 6/8)

# POSITIVE: 3.85 3.11 3.03 2.68 2.55 2.13 2.61 1.94
# NEGATIVE: 0.49  0.65 2.96 5.08 2.44  1.49 2.69 7.52 19.45 11.18 13.96


def output_meteors(update_info, stream, debug_mode):
    met_lst, drop_lst = update_info
    for met in met_lst:
        stream("Meteor:", met)
    if debug_mode:
        for met in drop_lst:
            stream("Dropped: ", met)


def detect_video(video_name,
                 mask_name,
                 cfg,
                 debug_mode,
                 work_mode="frontend",
                 time_range=(None, None)):
    
    t0 = time.time()
    
    # load config from cfg json.
    resize_param = cfg.resize_param
    meteor_cfg = cfg.meteor_cfg

    # set output mode
    progout = set_out_pipe(work_mode)

    # load video, init video_reader
    video, mask = load_video_and_mask(video_name, mask_name, resize_param)
    resize_param = list(reversed(mask.shape))
    progout("Apply running-time resolution = %s." % resize_param)
    
    # get accurate start_frame and end_frame according to the input arguments.
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
                                     merge_func=cfg.stacker_cfg["pfunc"])
    # Acquire exposure time and eqirvent FPS(eq_fps)
    total_frame, fps = video.num_frames, video.fps
    if cfg.stacker == "SimpleStacker":
        progout(
            "Ignore the option \"exp_time\" when appling \"SimpleStacker\".")
        exp_time, exp_frame, eq_fps, eq_int_fps = 1 / fps, 1, fps, int(fps)
    else:
        progout("Parsing \"exp_time\"=%s" % (cfg.exp_time))
        exp_time = init_exp_time(cfg.exp_time, video_reader, upper_bound=0.25)
        exp_frame, eq_fps, eq_int_fps = int(round(
            exp_time * fps)), 1 / exp_time, floor(1 / exp_time)
    progout("Apply exposure time of %.2fs. (MinTimeFlag = %d)" %
            (exp_time, (1000 * exp_frame * eq_int_fps / fps)))
    progout("Total frames = %d ; FPS = %.2f (rFPS = %.2f)" %
            (end_frame - start_frame, fps, eq_fps))
    progout(f"Preprocessing finished. Time cost: {(time.time() - t0):.1f}s.")
    # Reset video reader for main progress.
    video_reader.reset(start_frame = start_frame, iterations = end_frame - start_frame, exp_frame=exp_frame)

    # Init detector
    cfg.detect_cfg.update(img_mask=mask)
    detector = init_detector(cfg.detector, cfg.detect_cfg, eq_fps)

    # Init meteor collector
    # TODO: To be renewed
    # TODO: Update My Munch
    meteor_cfg = Munch(meteor_cfg)
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
    main_mc = MeteorCollector(**meteor_cfg, eframe=exp_frame, fps=fps)

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
            if work_mode == 'backend' and (((i - start_frame) // exp_frame) %
                                           eq_int_fps == 0):
                progout("Processing: %d" % (int(1000 * i / fps)))

            if video_reader.stopped and video_reader.is_empty:
                break
            
            detector.update(video_reader.pop())

            #TODO: Mask, visual
            lines, img_visu = detector.detect()

            if len(lines) or (((i - start_frame) // exp_frame) % eq_int_fps
                              == 0):
                output_meteors(main_mc.update(i, lines=lines), progout,
                               debug_mode)
            if debug_mode:
                if (cv2.waitKey(int(exp_time * 400)) & 0xff == ord("q")):
                    break
                # img_visu is used to get the image,
                # and is only executed when visualizing.
                draw_img = main_mc.draw_on_img(img_visu(), frame_num=i)
                #cv2.imwrite("test/frame_%s.jpg"%i,draw_img)
                cv2.imshow("Debug Window (Press Q to exit)", draw_img)

    finally:
        video_reader.stop()
        output_meteors(main_mc.update(np.inf, []), progout, debug_mode)
        video.release()
        cv2.destroyAllWindows()
        progout('Video EOF detected.')
        progout("Time cost: %.4ss." % (time.time() - t0))

    return resize_param, main_mc.ended_meteor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Meteor Detector V1.3')

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
                          help="Apply adaptive binary threshold.")

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
    # replace config value
    if exp_time:
        cfg.exp_time = exp_time
    if resize_param:
        cfg.resize_param = resize_param
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
