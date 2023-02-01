import argparse
import os

from MetLib.Stacker import max_stacker, time2frame
from MetLib.utils import Munch, save_img
from MetLib.VideoWarpper import OpenCVVideoWarpper

argparser = argparse.ArgumentParser()
argparser.add_argument("target", type=str, help="the target video.")
argparser.add_argument("--start-time",
                       type=str,
                       help="the start time of the clip.")
argparser.add_argument("--end-time",
                       type=str,
                       help="the end time of the clip.")
argparser.add_argument("--save-path",
                       type=str,
                       help="the path where the image is placed. \
            If the filename is not included, it will be automatically named.",
                       default=None)

args = argparser.parse_args()
video_name, t1, t2, img_path = args.target, args.start_time, args.end_time, args.save_path

if (img_path == None) or os.path.isdir(img_path):
    img_path_only, img_name = os.getcwd() if img_path == None else img_path, ""
else:
    img_path_only, img_name = os.path.split(img_path)

if img_name == "":
    _, video_name_nopath = os.path.split(video_name)
    video_name_pure = ".".join(video_name_nopath.split(".")[:-1])
    img_name = f"{video_name_pure}_{t1}-{t2}.jpg"
    img_name = img_name.replace(":", "_")
    img_path = os.path.join(img_path_only, img_name)

video = OpenCVVideoWarpper(video_name)
fps = video.fps
results = max_stacker(video, time2frame(t1, fps), time2frame(t2, fps))

save_img(results, img_path)
