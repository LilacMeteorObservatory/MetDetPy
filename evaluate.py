import argparse
import json
import os
from collections import namedtuple
import psutil

import numpy as np
from easydict import EasyDict
from MetDetPy import detect_video
from MetLib.utils import met2xyxy, ts2frame, calculate_area_iou, relative2abs_path, VERSION
from MetLib.VideoWrapper import OpenCVVideoWrapper
from typing import Any

import time
import threading


def monitor_performance(func, args: list, kwargs: dict, interval=0.5) -> tuple:
    """运行给定的函数，并统计运行期间的CPU和内存开销。

    Args:
        func (Callable): 待评估函数
        args (list): 参数列表
        kwargs (dict): 关键字参数列表
        interval (float, optional): 评估时间间隔. Defaults to 0.5.

    Returns:
        tuple: 一个元组，0位为效果，1位为返回值。
    """
    process = psutil.Process()
    start_time = time.time()
    cpu_samples = []
    memory_samples = []

    # 定义采样线程
    def sample():
        while not stop_event.is_set():
            cpu_samples.append(process.cpu_percent(interval=None))
            memory_samples.append(process.memory_info().rss)
            time.sleep(interval)

    # 创建并启动采样线程
    stop_event = threading.Event()
    sampling_thread = threading.Thread(target=sample)
    sampling_thread.start()

    try:
        # 执行被装饰的函数
        result = func(*args, **kwargs)
    finally:
        # 停止采样线程
        stop_event.set()
        sampling_thread.join()

    # 记录结束时间
    end_time = time.time()

    # 计算运行时长
    run_time = end_time - start_time

    # 计算平均CPU占用率
    avg_cpu_usage = sum(cpu_samples) / len(cpu_samples)

    # 计算平均内存开销
    avg_memory_usage = sum(memory_samples) / len(
        memory_samples) / 1024 / 1024  # 转换为MB

    # 打印统计信息
    stats = dict(tot_time=run_time,
                 avg_cpu_usage=avg_cpu_usage,
                 avg_mem_usage=avg_memory_usage)

    return stats, result


def get_regularized_results(result_dict,
                            video: OpenCVVideoWrapper) -> list[dict]:
    """从报告结果生成真实尺寸和帧时间表示下的结果列表.

    主要涉及到尺寸重放缩与时间戳转换

    Args:
        result_dict (_type_): _description_

    Returns:
        list[dict]: _description_
    """
    real_size = video.size
    fps = video.fps

    anno_size = getattr(result_dict, "anno_size", None)
    results = getattr(result_dict, "results", None)
    assert anno_size != None and results != None, \
            "Metrics can only be applied when \"anno_size\" and \"results\" are provided!"

    ax, ay = anno_size
    dx, dy = real_size
    scaler = dx / ax, dy / ay
    scale = lambda x: [i * s for (i, s) in zip(x, scaler)]

    for single_anno in results:
        single_anno["pt1"] = scale(single_anno["pt1"])
        single_anno["pt2"] = scale(single_anno["pt2"])
        single_anno["start_frame"] = ts2frame(single_anno["start_time"], fps)
        single_anno["end_frame"] = ts2frame(single_anno["end_time"], fps)
    return results


def calculate_time_iou(met_a, met_b):
    """计算时间ioU.

    Args:
        met_a (_type_): _description_
        met_b (_type_): _description_

    Returns:
        _type_: _description_
    """
    if (met_a["start_frame"]
            >= met_b["end_frame"]) or (met_a["end_frame"]
                                       <= met_b["start_frame"]):
        return 0
    t = sorted([
        met_a["start_frame"], met_a["end_frame"], met_b["start_frame"],
        met_b["end_frame"]
    ],
               reverse=True)
    return (t[1] - t[2]) / (t[0] - t[3])


def compare_with_annotation():
    pass


def compare(video: OpenCVVideoWrapper,
            base_dict,
            new_dict,
            pos_thre=0.5,
            tiou=0.3,
            aiou=0.3):
    """比较两个结果。

    与其他运行结果比较：
    性能部分：
    1. cpu占用情况
    2. 运行时间
    3. 平均内存开销
    效果部分：
    1. 预测样本相交率
    2. 相交样本的平均离差

    与GT比较：
    1. 准确率
    2. 召回率
    3. F1-Score

    Args:
        base_dict (_type_): _description_
        new_dict (_type_): _description_
    """
    gt_mode = (base_dict.type == "annotation")

    # TODO: 分别计算长/中/短的P/R/F1（长中短的划分如何决定？）
    # List of Gts

    base_results = get_regularized_results(base_dict, video)
    new_results = get_regularized_results(new_dict, video)

    # 主要指标
    # True Positive / False Positive（误报） / False Negative（漏报）
    tp, fp, fn = 0, 0, 0
    gt_id = 0
    end_flag = False

    tp_list = []
    fp_list = []
    fn_list = []

    matched_pair_list = []
    matched_id = np.zeros((len(base_results), ), dtype=bool)

    # 正样本阈值：默认0.5
    # 匹配要求：TIoU threshold=0.3(??) & IoU threshold=0.3 且具有唯一性(?)
    for i, instance in enumerate(new_results):
        # 只在与Ground Truth对比时需要过滤非置信（得分低于正样本阈值）的预测
        if gt_mode and instance["score"] <= pos_thre:
            continue

        # 向后更新gt_id
        # move gt_id to the next possible match
        while instance["start_time"] >= base_results[gt_id]["end_time"]:
            gt_id += 1
            if gt_id == len(base_results):
                end_flag = True
                break
        if end_flag:
            break

        # 为当前instance向后查找是否存在匹配
        match_flag = False
        cur_id = gt_id
        while instance["end_time"] >= base_results[cur_id]["start_time"]:
            if matched_id[cur_id] == 0 \
                and (calculate_time_iou(instance,base_results[cur_id]) >= tiou) \
                and calculate_area_iou(met2xyxy(instance), met2xyxy(base_results[cur_id])) >= aiou:
                match_flag = True
                tp += 1
                matched_id[cur_id] = 1
                matched_pair_list.append([i, cur_id])
                break
            cur_id += 1
            if cur_id == len(base_results):
                match_flag = False
                break
        if not match_flag:
            fp += 1

    new_predict_num = len(new_results)
    old_predict_num = len(base_results)
    tp_num = np.sum(matched_id == 1)
    #fn_list = np.array(base_results)[matched_id == 0]
    fn_num = old_predict_num - tp_num
    tn_num = new_predict_num - tp_num

    compare_result = {
        "matched_num":
        tp_num,
        "new_predict_num":
        new_predict_num,
        "old_predict_num":
        old_predict_num,
        "cross_ratio(A n B / A u B)":
        tp_num / (new_predict_num + old_predict_num - tp_num),
        "fn_num":
        fn_num,
        "tn_num":
        tn_num
    }

    import pprint
    pprint.pprint(compare_result)

    #print(
    #    f"True Positive = {tp}; False Positive = {fp}; False Negative = {fn};")
    #print(
    #    f"Precision = {tp/(tp+fp)*100:.2f}%; Recall = {tp/(tp+fn)*100:.2f}%; ")
    #print(np.array(gt_meteors)[matched_id==0][:10])


def generate_result(video, raw_basic_info, cfg, performance,
                    results: list) -> EasyDict:
    """根据检测结果生成报告字典。

    Args:
        results (list): _description_

    Returns:
        dict: _description_
    """
    # 构造结果信息中的基础部分
    result_basic_info = raw_basic_info
    if not result_basic_info.get("fps", None):
        result_basic_info.fps = video.fps
    if not result_basic_info.get("desc", None):
        result_basic_info.desc = "待检测视频的基础信息 | Basic infomation about the video"
    # 补充performance部分
    performance["desc"] = "硬件指标 | Hardware performance"
    performance["cpu_core"] = psutil.cpu_count(logical=True)
    # TODO: 调用接口获取的分辨率仍然是960x540下的。这个需要在未来更正。紧急！
    return EasyDict(
        version=VERSION,
        basic_info=result_basic_info,
        config=cfg,
        performance=performance,
        type="prediction",
        anno_size=[960, 540],  #video.size,
        results=results)


def main():
    # 可选模式
    # 1. 生成报告：对视频片段进行检测，给出当前版本下给定配置的报告。
    #    （视频片段的信息从给定的报告/GroundTruth中摘录得到）
    # 2. 效果回归：对当前视频与参考报告的检测效果和内存开销对比。
    #      a) 使用--load选项时，load选项作为当前的主结果。
    #      b) 当与annotation比较时，相当于计算检测指标；否则按照回归测试。
    # 3. TODO: 批处理：对一批数据执行类似操作。
    parser = argparse.ArgumentParser(description='MetDetPy Evaluater.')

    parser.add_argument('json', help="json file of test videos.")

    parser.add_argument(
        '--cfg',
        '-C',
        help="Config file.",
        default=relative2abs_path("./config/m3det_normal.json"))

    parser.add_argument(
        '--load',
        '-L',
        help="Load a result file instead of running on datasets.",
        default=None)

    parser.add_argument('--save',
                        '-S',
                        help="Save a result files.",
                        default=None)

    parser.add_argument('--metric',
                        '-M',
                        action="store_true",
                        help="Calculate metrics with the base json",
                        default=False)

    parser.add_argument('--debug',
                        '-D',
                        action='store_true',
                        help="Apply Debug Mode",
                        default=False)

    args = parser.parse_args()

    ## Load video and config
    with open(args.json, mode='r', encoding='utf-8') as f:
        video_dict: Any = EasyDict(json.load(f))

    with open(args.cfg, mode='r', encoding='utf-8') as f:
        cfg = EasyDict(json.load(f))

    video_name = video_dict.basic_info.video
    mask_name = video_dict.basic_info.mask
    start_time = video_dict.basic_info.start_time
    end_time = video_dict.basic_info.end_time

    # 对于json文件放置在video/mask同路径下的，使用共享的相对路径
    shared_path = os.path.split(args.json)[0]
    if os.path.split(video_name)[0] == "":
        video_name = os.path.join(shared_path, video_name)
        video_dict.basic_info.video = video_name
    if (mask_name != "") and (os.path.split(mask_name)[0] == ""):
        mask_name = os.path.join(shared_path, mask_name)
        video_dict.basic_info.mask = mask_name

    video = OpenCVVideoWrapper(video_name)
    try:
        if args.load:
            with open(args.load, mode='r', encoding="utf-8") as f:
                new_result = EasyDict(json.load(f))
        else:
            performance, results = monitor_performance(
                detect_video, [video_name, mask_name, cfg, args.debug],
                dict(work_mode="frontend", time_range=(start_time, end_time)))
            new_result = generate_result(video,
                                         raw_basic_info=video_dict.basic_info,
                                         cfg=cfg,
                                         performance=performance,
                                         results=results)
            if args.save:
                # List of predictions
                with open(args.save, mode='w', encoding="utf-8") as f:
                    json.dump(new_result, f)

        if args.metric:
            compare(video, base_dict=video_dict, new_dict=new_result)
    finally:
        video.release()


if __name__ == "__main__":
    main()