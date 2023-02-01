import argparse
import json
import os

from MetLib.Stacker import max_stacker, all_stacker, time2frame
from MetLib.utils import save_img, save_video
from MetLib.VideoWarpper import OpenCVVideoWarpper


def stack_and_save_img(video, start_time, end_time, path):
    results = max_stacker(video, time2frame(start_time, video.fps),
                          time2frame(end_time, video.fps))
    save_img(results, path)


def clip_and_save_video(video, start_time, end_time, path):
    video_series = all_stacker(video, time2frame(start_time, video.fps),
                               time2frame(end_time, video.fps))
    save_video(video, video_series, path)


argparser = argparse.ArgumentParser()
argparser.add_argument("target", type=str, help="the target video.")
argparser.add_argument(
    "json",
    type=str,
    default=None,
    help=
    "a json-format string or the path to a json file where start-time and end-time are listed."
)
argparser.add_argument("--mode",
                       choices=['image', 'video'],
                       default='image',
                       type=str,
                       help="convert clip to video or image.")
argparser.add_argument("--suffix",
                       type=str,
                       help="the suffix of the output. \
                           By default, it is \"jpg\" for img mode and \"avi\" for video mode.",
                       default=None)
argparser.add_argument("--save-path",
                       type=str,
                       help="the path where image(s)/video(s) are placed. Only\
                           path should be provided: they will be automatically named.",
                       default=os.getcwd())

mode2func = {"image": stack_and_save_img, "video": clip_and_save_video}

args = argparser.parse_args()
video_name, json_str, mode, suffix, save_path = args.target, args.json, args.mode, args.suffix, args.save_path
video = OpenCVVideoWarpper(video_name)
fps = video.fps

# parse json argument
data = None
if os.path.isfile(json_str):
    with open(json_str, mode='r', encoding='utf-8') as f:
        data = json.load(f)
else:
    data = json.loads(json_str)

# get video name
_, video_name_nopath = os.path.split(video_name)
video_name_pure = ".".join(video_name_nopath.split(".")[:-1])
# get correct suffix
if suffix is None:
    if mode == "image": suffix = "jpg"
    if mode == "video": suffix = "avi"
main_func = mode2func[mode]

for time_pair in data:
    start_time, end_time = time_pair

    tgt_name = f"{video_name_pure}_{start_time}-{end_time}.{suffix}"
    tgt_name = tgt_name.replace(":", "_")
    full_path = os.path.join(save_path, tgt_name)

    main_func(video, start_time, end_time, full_path)