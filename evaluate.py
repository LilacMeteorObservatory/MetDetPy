import argparse
import json
import os
import threading
import time
from typing import Any, Callable, TypeVar, Union
from numpy.typing import NDArray
import numpy as np
import psutil

from MetDetPy import detect_video
from MetLib.fileio import save_path_handler
from MetLib.metstruct import (MDRF, BasicInfo, MainDetectCfg, MDTarget,
                              MockVideoObject, SingleMDRecord)
from MetLib.utils import (NAME2ID, NUM_CLASS, calculate_area_iou, met2xyxy,
                          relative2abs_path)
from MetLib.videowrapper import OpenCVVideoWrapper

T = TypeVar("T")


def scale(x: list[int], scaler: list[float]):
    return [int(i * s) for (i, s) in zip(x, scaler)]


def monitor_performance(func: Callable[..., T],
                        args: list[Any],
                        kwargs: dict[str, Any],
                        interval: float = 0.5) -> tuple[dict[str, float], T]:
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
    cpu_samples: list[float] = []
    memory_samples: list[float] = []

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


def get_regularized_results(result_dict: MDRF,
                            video: OpenCVVideoWrapper) -> list[MDTarget]:
    """从报告结果生成真实尺寸和帧时间表示下的结果列表.

    主要涉及到尺寸重放缩与时间戳转换

    Args:
        result_dict (_type_): _description_

    Returns:
        list[dict]: _description_
    """
    real_size = video.size

    anno_size = result_dict.anno_size
    results = result_dict.results
    assert anno_size != None and results != None, \
            "Metrics can only be applied when \"anno_size\" and \"results\" are provided!"
    results_flatten = [
        target for x in results if isinstance(x, SingleMDRecord)
        for target in x.target
    ]
    ax, ay = anno_size
    dx, dy = real_size
    scaler = [dx / ax, dy / ay]

    for single_anno in results_flatten:
        single_anno.pt1 = scale(single_anno.pt1, scaler)
        single_anno.pt2 = scale(single_anno.pt2, scaler)
    return results_flatten


def calculate_time_iou(met_a: MDTarget, met_b: MDTarget):
    """计算时间ioU.

    Args:
        met_a (_type_): _description_
        met_b (_type_): _description_

    Returns:
        _type_: _description_
    """
    #last_activate_frame
    if (met_a.start_frame
            >= met_b.last_activate_frame) or (met_a.last_activate_frame
                                              <= met_b.start_frame):
        return 0
    t = sorted([
        met_a.start_frame, met_a.last_activate_frame, met_b.start_frame,
        met_b.last_activate_frame
    ],
               reverse=True)
    return (t[1] - t[2]) / (t[0] - t[3])


def compare_with_annotation():
    pass


def print_confusion_matrix(matrix: NDArray[np.int_], labels: list[str]):
    """
    打印混淆矩阵的纯文本表格
    Args:
        matrix: ndarray, shape (N, N)
        labels: list, 标签列表
    """
    # 计算每列宽度
    head_col_width = 15
    col_width = 5

    # 构建表头
    header = 'PRED\\BASE'.center(head_col_width) + '|'
    header += ''.join(label[:col_width].center(col_width) + '|'
                      for label in labels)
    separator = '-' * head_col_width + '+'
    separator += '+'.join('-' * col_width for _ in labels)

    # 打印表头
    print(header)
    print(separator)

    # 打印每一行
    for i, label in enumerate(labels):
        row = label.ljust(head_col_width) + '|'
        row += ''.join(str(cell).center(col_width) + '|' for cell in matrix[i])
        print(row)
        print(separator)


def compare(video: OpenCVVideoWrapper,
            base_dict: MDRF,
            new_dict: MDRF,
            pos_thre: float = 0.5,
            tiou: float = 0.3,
            aiou: float = 0.3) -> MDRF:
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
    
    Return:
        返回所有错配的结果...
    """
    gt_mode = (base_dict.type == "annotation")

    # TODO: 分别计算长/中/短的P/R/F1（长中短的划分如何决定？）
    # List of Gts

    base_results = get_regularized_results(base_dict, video)
    new_results = get_regularized_results(new_dict, video)

    mismatch_collection: list[MDTarget] = []

    # 主要指标
    # True Positive / False Positive（误报） / False Negative（漏报）
    tp, fp = 0, 0
    gt_id = 0
    end_flag = False

    confusion_matrix = np.zeros((NUM_CLASS + 1, NUM_CLASS + 1), dtype=np.int16)

    matched_pair_list: list[tuple[int, int]] = []
    matched_id = np.zeros((len(base_results), ), dtype=bool)

    # 正样本阈值：默认0.5
    # 匹配要求：TIoU threshold=0.3(??) & IoU threshold=0.3 且具有唯一性(?)
    for i, instance in enumerate(new_results):
        # 只在与Ground Truth对比时需要过滤非置信（得分低于正样本阈值）的预测
        if gt_mode and instance.score <= pos_thre:
            continue

        # 向后更新gt_id
        # move gt_id to the next possible match
        while instance.start_time >= base_results[gt_id].end_time:
            gt_id += 1
            if gt_id == len(base_results):
                end_flag = True
                break
        if end_flag:
            break

        # 为当前instance向后查找是否存在匹配
        match_flag = False
        cur_id = gt_id
        while instance.end_time >= base_results[cur_id].start_time:
            if matched_id[cur_id] == 0 \
                and (calculate_time_iou(instance,base_results[cur_id]) >= tiou) \
                and calculate_area_iou(met2xyxy(instance.to_dict()), met2xyxy(base_results[cur_id].to_dict())) >= aiou:
                # TEMP FIX: 向前兼容v2.1.0的标注，低置信度转DROPPED进行判定。
                if base_results[cur_id].score <= pos_thre:
                    base_results[cur_id].category = "DROPPED"
                base_category = base_results[cur_id].category
                # 兼容。。。
                if base_category == "UNKNOWN_AREA":
                    base_category = "OTHERS"
                confusion_matrix[NAME2ID[instance.category],
                                 NAME2ID[base_category]] += 1
                if NAME2ID[instance.category] != NAME2ID[base_category]:
                    mismatch_collection.append(instance)
                match_flag = True
                tp += 1
                matched_id[cur_id] = 1
                matched_pair_list.append((i, cur_id))
                break
            cur_id += 1
            if cur_id == len(base_results):
                match_flag = False
                break
        if not match_flag:
            confusion_matrix[NAME2ID[instance.category], -1] += 1
            fp += 1

    new_predict_num = len(new_results)
    old_predict_num = len(base_results)
    tp_num = np.sum(matched_id == 1)
    #fn_list = np.array(base_results)[matched_id == 0]
    fn_num = old_predict_num - tp_num
    tn_num = new_predict_num - tp_num

    compare_result: dict[str, Union[int, float]] = {
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
    print_confusion_matrix(confusion_matrix, list(NAME2ID.keys()) + ["MISSED"])

    import copy
    return_dict = copy.deepcopy(new_dict)
    assert new_dict.anno_size is not None, "Invalid anno size..."
    return_dict.results = [
        SingleMDRecord.from_target(x, new_dict.anno_size)
        for x in mismatch_collection
    ]
    return return_dict

    #print(
    #    f"True Positive = {tp}; False Positive = {fp}; False Negative = {fn};")
    #print(
    #    f"Precision = {tp/(tp+fp)*100:.2f}%; Recall = {tp/(tp+fn)*100:.2f}%; ")
    #print(np.array(gt_meteors)[matched_id==0][:10])


def generate_full_result(results: MDRF,
                         performance: dict[str, Union[float, str, None]]):
    # 补充必要信息
    assert isinstance(results.basic_info, BasicInfo), "Invalid basic info!"
    results.basic_info.desc = "待检测视频的基础信息 | Basic infomation about the video"
    performance["desc"] = "硬件指标 | Hardware performance"
    performance["cpu_core"] = psutil.cpu_count(logical=True)
    results.performance = performance
    return results


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

    parser.add_argument('--save-path',
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
    video_dict = MDRF.from_json_file(args.json)
    cfg = MainDetectCfg.from_json_file(args.cfg)
    # 暂时不支持比对图像检测结果。
    if video_dict.basic_info is None or isinstance(video_dict.basic_info,
                                                   MockVideoObject):
        return
    video_name = video_dict.basic_info.video
    mask_name = video_dict.basic_info.mask
    start_time = video_dict.basic_info.start_time
    end_time = video_dict.basic_info.end_time

    # 对于json文件放置在video/mask同路径下的，使用共享的相对路径
    shared_path: str = os.path.split(args.json)[0]
    if os.path.split(video_name)[0] == "":
        video_name = os.path.join(shared_path, video_name)
        video_dict.basic_info.video = video_name
    if (mask_name) and (os.path.split(mask_name)[0] == ""):
        mask_name = os.path.join(shared_path, mask_name)
        video_dict.basic_info.mask = mask_name

    video = OpenCVVideoWrapper(video_name)
    try:
        if args.load:
            new_result = MDRF.from_json_file(args.load)
        else:
            performance, results = monitor_performance(
                detect_video, [video_name, mask_name, cfg, args.debug],
                dict(work_mode="frontend",
                     time_range=(str(start_time), str(end_time))))
            # 补充performance信息
            new_result = generate_full_result(results,
                                              performance)  # type: ignore
            if args.save_path:
                # List of predictions
                save_path = save_path_handler(args.save_path,
                                              video_name,
                                              ext="json")
                with open(save_path, mode='w', encoding="utf-8") as f:
                    json.dump(new_result.to_dict(),
                              f,
                              ensure_ascii=False,
                              indent=4)

        if args.metric:
            mismatch = compare(video,
                               base_dict=video_dict,
                               new_dict=new_result)
            with open("mismatch.json", mode="w", encoding="utf-8") as f:
                json.dump(mismatch.to_dict(), f, ensure_ascii=False, indent=4)
    finally:
        video.release()


if __name__ == "__main__":
    main()
