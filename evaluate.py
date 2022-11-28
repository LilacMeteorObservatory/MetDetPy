import argparse
import json
import os

from core import detect_video
from MetLib.utils import Munch


def resize_gt_coord(gts, anno_size, detect_size):
    """将标注中的坐标转换为实际坐标下的位置，便于计算IoU。

    Args:
        gts (list): 标注列表
        anno_size (_type_): _description_
        detect_size (_type_): _description_
    """
    ax, ay = anno_size
    dx, dy = detect_size
    scaler = dx / ax, dy / ay
    scale = lambda x: [i * s for (i, s) in zip(x, scaler)]

    for anno in gts:
        anno["pt1"] = scale(anno["pt1"])
        anno["pt2"] = scale(anno["pt2"])

    return gts


parser = argparse.ArgumentParser(description='MetDetPy Evaluater.')

parser.add_argument('video_json', help="json file of test videos.")

parser.add_argument('--cfg',
                    '-C',
                    help="Config file.",
                    default="./config.json")

parser.add_argument('--debug',
                    '-D',
                    action='store_true',
                    help="Apply Debug Mode",
                    default=False)

args = parser.parse_args()

## Load video and config

with open(args.video_json, mode='r', encoding='utf-8') as f:
    video_dict = Munch(json.load(f))

with open('config.json', mode='r', encoding='utf-8') as f:
    cfg = Munch(json.load(f))

##  我想了下。。感觉大多数微调检测参数的指令也需要带着。。
##  有关Argparse的部分大概会写的很混沌吧。
##  暂时不加了先

video_name = video_dict.video
mask_name = video_dict.mask

shared_path = os.path.split(args.video_json)[0]
if os.path.split(video_name)[0] == "":
    video_name = os.path.join(shared_path, video_name)
if os.path.split(mask_name)[0] == "":
    mask_name = os.path.join(shared_path, mask_name)

start_time = getattr(video_dict, "start_time", None)
end_time = getattr(video_dict, "end_time", None)

resize, results = detect_video(video_name,
                               mask_name,
                               cfg,
                               args.debug,
                               work_mode="frontend",
                               time_range=(start_time, end_time))
# List of Gts
meteor = resize_gt_coord(video_dict.meteors, video_dict.anno_size, resize)

# List of predictions
results