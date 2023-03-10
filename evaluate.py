import argparse
import json
import os
from collections import namedtuple

import numpy as np

from core import detect_video
from MetLib.utils import Munch, time2frame
from MetLib.VideoWarpper import OpenCVVideoWarpper
from ClipToolkit import stack_and_save_img

# 正样本阈值：默认0.5
# 匹配要求：TIoU threshold=0.3(??) & IoU threshold=0.3 且具有唯一性(?)
pos_thre = 0.5
tiou = 0.05
aiou = 0.3

box = namedtuple("box", ["x1", "y1", "x2", "y2"])


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
    #print(scaler)
    scale = lambda x: [i * s for (i, s) in zip(x, scaler)]

    for anno in gts:
        anno["pt1"] = scale(anno["pt1"])
        anno["pt2"] = scale(anno["pt2"])
    return gts


def batch_time2frame(meteors, fps):
    for meteor in meteors:
        meteor["start_frame"] = time2frame(meteor["start_time"], fps)
        meteor["end_frame"] = time2frame(meteor["end_time"], fps)
    return meteors


def met2xyxy(met):
    """将met的字典转换为xyxy形式的坐标。

    Args:
        met (_type_): _description_
    """
    (x1, y1), (x2, y2) = met["pt1"], met["pt2"]
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    return box(x1, y1, x2, y2)


def calculate_time_iou(met_a, met_b):
    if (met_a["start_frame"] >= met_b["end_frame"]) or (met_a["end_frame"] <=
                                                        met_b["start_frame"]):
        return 0
    t = sorted([
        met_a["start_frame"], met_a["end_frame"], met_b["start_frame"],
        met_b["end_frame"]
    ],
               reverse=True)
    return (t[1] - t[2]) / (t[0] - t[3])


def calculate_area_iou(met_a, met_b):
    """用于计算面积的iou。

    Args:
        met_a (_type_): _description_
        met_b (_type_): _description_
    """
    mat1 = met2xyxy(met_a)
    mat2 = met2xyxy(met_b)
    print(mat1, mat2)
    # 若无交集即为0
    if (mat1.x1 >= mat2.x2 or mat1.x2 <= mat2.x1) or (mat1.y1 >= mat2.y2
                                                      or mat1.y2 <= mat2.y1):
        return 0

    # 计算交集面积
    i_xx = sorted([mat1.x1, mat1.x2, mat2.x1, mat2.x2], reverse=True)[1:-1]
    i_yy = sorted([mat1.y1, mat1.y2, mat2.y1, mat2.y2], reverse=True)[1:-1]
    area_i = (i_xx[1] - i_xx[0]) * (i_yy[1] - i_yy[0])

    # 分别计算面积
    area_a = (mat1.x2 - mat1.x1) * (mat1.y2 - mat1.y1)
    area_b = (mat2.x2 - mat2.x1) * (mat2.y2 - mat2.y1)
    print(area_i, area_a, area_b)
    return area_i / (area_a + area_b - area_i)


parser = argparse.ArgumentParser(description='MetDetPy Evaluater.')

parser.add_argument('video_json', help="json file of test videos.")

parser.add_argument('--cfg',
                    '-C',
                    help="Config file.",
                    default="./config.json")

parser.add_argument('--load',
                    '-L',
                    help="Load a result file instead of running on datasets.",
                    default=None)

parser.add_argument('--save', '-S', help="Save a result files.", default=None)

parser.add_argument('--metrics',
                    '-M',
                    action='store_true',
                    help="Calculate metrics",
                    default=False)

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

video = OpenCVVideoWarpper(video_name)
raw_size = video.size
fps = video.fps
#video.release()

anno_size = getattr(video_dict, "anno_size", None)
gt_meteors = getattr(video_dict, "meteors", None)
start_time = getattr(video_dict, "start_time", None)
end_time = getattr(video_dict, "end_time", None)

shared_path = os.path.split(args.video_json)[0]
if os.path.split(video_name)[0] == "":
    video_name = os.path.join(shared_path, video_name)
if (mask_name!="") and (os.path.split(mask_name)[0] == ""):
    mask_name = os.path.join(shared_path, mask_name)

if args.load:
    with open(args.load, mode='r', encoding="utf-8") as f:
        results = json.load(f)
else:
    results = detect_video(video_name,
                           mask_name,
                           cfg,
                           args.debug,
                           work_mode="frontend",
                           time_range=(start_time, end_time))
    if args.save:
        # List of predictions
        with open(args.save, mode='w', encoding="utf-8") as f:
            json.dump(results, f)

# Precision（准确率）/Recall（召回率）/ F1-Score
# 👆并分别计算长/中/短的P/R/F1
# （长中短的划分如何决定？）

# List of Gts
if args.metrics:
    assert anno_size != None and gt_meteors != None, \
        "Metrics can only be applied when \"anno_size\" and \"meteors\" are provided!"
    gt_meteors = resize_gt_coord(video_dict.meteors, anno_size, raw_size)
    results = resize_gt_coord(results, [960, 540], raw_size)
    gt_meteors = batch_time2frame(gt_meteors, fps)
    results = batch_time2frame(results, fps)
    # 主要指标
    # True Positive / False Positive（误报） / False Negative（漏报）
    tp, fp, fn = 0, 0, 0
    gt_id = 0
    end_flag = False

    tp_list = []
    fp_list = []
    fn_list = []

    gt_label = np.zeros((len(gt_meteors), ), dtype=bool)

    for instance in results:
        if instance["score"] <= pos_thre:
            continue
        # move gt_id to the next possible match
        # Notice: time should be resize to `float, int`.
        while instance["start_time"] >= gt_meteors[gt_id]["end_time"]:
            gt_id += 1
            if gt_id == len(gt_meteors):
                end_flag = True
                break
        if end_flag:
            break
        match_flag = False
        cur_id = gt_id
        while instance["end_time"] >= gt_meteors[cur_id]["start_time"]:
            if gt_label[cur_id] == 0 and (calculate_time_iou(
                    instance, gt_meteors[cur_id]) >= tiou):
                # and calculate_area_iou(
                #        instance, gt_meteors[cur_id]) >= aiou
                match_flag = True
                tp += 1
                gt_label[cur_id] = 1
                break
            cur_id += 1
            if cur_id == len(gt_meteors):
                match_flag = False
                break
        if not match_flag:
            fp += 1

    fn_list = np.array(gt_meteors)[gt_label == 0]

    for i in range(10):
        instance=fn_list[i]
        print(i,instance)
        stack_and_save_img(video, instance["start_time"], instance["end_time"], f"./fn_{i}.jpg",85,3,None)


    fn = len(gt_meteors) - tp

    print(
        f"True Positive = {tp}; False Positive = {fp}; False Negative = {fn};")
    print(
        f"Precision = {tp/(tp+fp)*100:.2f}%; Recall = {tp/(tp+fn)*100:.2f}%; ")
    #print(np.array(gt_meteors)[gt_label==0][:10])