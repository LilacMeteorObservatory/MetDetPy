import argparse
import json
import os

from MetLib.Stacker import max_stacker, all_stacker, time2frame
from MetLib.utils import save_img, save_video
from MetLib.VideoWarpper import OpenCVVideoWarpper

support_image_suffix = ["JPG", "JPEG", "PNG"]
support_video_suffix = ["AVI"]


def stack_and_save_img(video, start_time, end_time, path, quality,
                       compressing):
    results = max_stacker(video, time2frame(start_time, video.fps),
                          time2frame(end_time, video.fps))
    save_img(results, path, quality, compressing)


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
                       help="convert clip to video or image.\
                           This option will be covered by specific filename in json."
                       )
argparser.add_argument("--suffix",
                       type=str,
                       help="the suffix of the output. \
                           By default, it is \"jpg\" for img mode and \"avi\" for video mode.\
                           This option will be covered by specific filename in json.",
                       default=None)
argparser.add_argument(
    "--save-path",
    type=str,
    help="the path where image(s)/video(s) are placed. When only ",
    default=os.getcwd())
argparser.add_argument(
    "--jpg-quality",
    type=int,
    help=
    "the quality of generated jpg image. It should be int ranged Z \in [0,100];\
        By default, it is 95.  This option only works for \"image\" mode.",
    default=95)

argparser.add_argument(
    "--png-compressing",
    type=int,
    help=
    "the compressing of generated png image. It should be int ranged Z \in [0,9];\
        By default, it is 3.  This option only works for \"image\" mode.",
    default=3)

argparser.add_argument(
    "--resize",
    type=str,
    help=
    "resize img/video to the given solution. (support img only now)",
    default=None)

mode2func = {"image": stack_and_save_img, "video": clip_and_save_video}

args = argparser.parse_args()
video_name, json_str, mode, default_suffix, save_path,  = \
    args.target, args.json, args.mode, args.suffix, args.save_path,

jpg_quality, png_compress = args.jpg_quality, args.png_compressing

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

# get default suffix
if default_suffix is None:
    if mode == "image": default_suffix = "jpg"
    if mode == "video": default_suffix = "avi"

# convert image quality to correct

# 单一片段时，若保存路径中包含文件名，覆盖输出的文件名。
if len(data) == 1 and (not os.path.isdir(save_path)):
    save_path, filename = os.path.split(save_path)
    data[0]["filename"] = filename

for single_data in data:
    start_time, end_time = single_data["time"]
    # 如果未给定名称则使用缺省名称
    tgt_name = single_data.get(
        "filename",
        f"{video_name_pure}_{start_time}-{end_time}.{default_suffix}")
    tgt_name = tgt_name.replace(":", "_")

    # 获取后缀，检查合法性
    cur_mode = mode
    suffix = tgt_name.split(".")[-1].upper()
    if suffix in support_image_suffix: cur_mode = "image"
    elif suffix in support_video_suffix: cur_mode = "video"
    else:
        print(f"Unsupport suffix: {suffix}. Ignore error and continue.")
        continue

    full_path = os.path.join(save_path, tgt_name)

    if cur_mode == "image":
        stack_and_save_img(video,
                           start_time,
                           end_time,
                           full_path,
                           quality=jpg_quality,
                           compressing=png_compress)
    else:
        clip_and_save_video(video, start_time, end_time, full_path)