import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
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
from MetLib.utils import (calculate_area_iou, met2xyxy, relative2abs_path,
                          get_name2id, get_num_class)
from MetLib.videowrapper import OpenCVVideoWrapper

T = TypeVar("T")
NAME2ID = get_name2id()
NUM_CLASS = get_num_class()


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
    if interval <= 0:
        raise ValueError("interval must be positive")
    process = psutil.Process()
    memory_samples = [process.memory_info().rss]
    stop_event = threading.Event()

    # 定义采样线程
    def sample():
        while not stop_event.wait(interval):
            memory_samples.append(process.memory_info().rss)

    sampling_thread = threading.Thread(target=sample)
    sampling_thread.start()
    cpu_start = process.cpu_times()
    start_time = time.perf_counter()
    try:
        result = func(*args, **kwargs)
    finally:
        run_time = time.perf_counter() - start_time
        cpu_end = process.cpu_times()
        stop_event.set()
        sampling_thread.join()
    memory_samples.append(process.memory_info().rss)
    cpu_time = (cpu_end.user + cpu_end.system - cpu_start.user - cpu_start.system)
    memory_mb = np.asarray(memory_samples, dtype=float) / (1024 * 1024)
    num_kernels = os.cpu_count()
    stats = dict(tot_time=run_time, cpu_time=cpu_time,
                 avg_cpu_usage=100 * cpu_time / run_time / num_kernels if run_time and num_kernels else 0.0,
                 avg_mem_usage=float(memory_mb.mean()),
                 peak_mem_usage=float(memory_mb.max()),
                 start_mem_usage=float(memory_mb[0]),
                 end_mem_usage=float(memory_mb[-1]),
                 mem_growth=float(memory_mb[-1] - memory_mb[0]),
                 mem_sample_count=len(memory_samples),
                 mem_sample_interval=interval)
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
        copy.deepcopy(target) for x in results if isinstance(x, SingleMDRecord)
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


def calculate_detection_metrics(base_results, new_results, pos_thre=0.5,
                                tiou=0.3, aiou=0.3, gt_mode=True):
    """分类指标及 PR 数据。先按分数降序做类别无关的一对一时空匹配。

    DROPPED 不参与分类指标；GT 中低分标注沿用 DROPPED 约定。
    零分母的 P/R/F1 返回 0，macro 只汇总实际出现的类别。
    """
    def category(item):
        return "OTHERS" if item.category == "UNKNOWN_AREA" else item.category

    references = [x for x in base_results if category(x) != "DROPPED"
                  and (not gt_mode or x.score > pos_thre)]
    predictions = sorted((x for x in new_results
                          if category(x) != "DROPPED"),
                         key=lambda x: x.score, reverse=True)
    labels = [name for name in NAME2ID if name != "DROPPED"]
    # 阈值扫描复用时空交叠计算，避免每个阈值重复计算 IoU。
    candidates = []
    for prediction in predictions:
        overlaps = []
        for index, reference in enumerate(references):
            time_iou = calculate_time_iou(prediction, reference)
            if time_iou < tiou:
                continue
            area_iou = calculate_area_iou(met2xyxy(prediction.to_dict()),
                                          met2xyxy(reference.to_dict()))
            if area_iou >= aiou:
                overlaps.append((time_iou * area_iou, index))
        candidates.append(sorted(overlaps, key=lambda pair: (-pair[0], pair[1])))

    def summarize(tp, fp, fn):
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall,
                    f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)

    def at_threshold(threshold):
        counts = {name: [0, 0, 0] for name in labels}
        matched = set()
        evaluated = 0
        for prediction, overlaps in zip(predictions, candidates):
            if prediction.score <= threshold:
                continue
            evaluated += 1
            pred_category = category(prediction)
            match = next((index for _, index in overlaps if index not in matched), None)
            if match is None:
                counts[pred_category][1] += 1
                continue
            matched.add(match)
            ref_category = category(references[match])
            if pred_category == ref_category:
                counts[pred_category][0] += 1
            else:
                counts[pred_category][1] += 1
                counts[ref_category][2] += 1
        for index, reference in enumerate(references):
            if index not in matched:
                counts[category(reference)][2] += 1
        per_class = {name: summarize(*values) for name, values in counts.items()}
        active = [value for value in per_class.values()
                  if value["tp"] + value["fp"] + value["fn"]]
        micro = summarize(*(sum(values[i] for values in counts.values())
                            for i in range(3)))
        macro = {key: sum(value[key] for value in active) / len(active) if active else 0.0
                 for key in ("precision", "recall", "f1")}
        return dict(threshold=threshold, matched_num=len(matched),
                    evaluated_predict_num=evaluated, per_class=per_class,
                    micro=micro, macro=macro)

    scores = [float(x.score) for x in predictions]
    thresholds = sorted(set(scores + [pos_thre] +
                            ([float(np.nextafter(min(scores), -np.inf))] if scores else [])),
                        reverse=True)
    return dict(mode="ground_truth" if gt_mode else "baseline_consistency",
                candidate_num=len(new_results),
                low_score_predict_num=sum(x.score <= pos_thre for x in new_results),
                dropped_predict_num=sum(category(x) == "DROPPED" for x in new_results),
                reference_num=len(references), metrics=at_threshold(pos_thre),
                pr_curve=[at_threshold(threshold) for threshold in thresholds])


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


def summarize_comparison(matrix):
    """矩阵行为预测类别、列为参考类别，末行/列表示未匹配。"""
    transitions = [dict(from_category=reference, to_category=prediction,
                        count=int(matrix[pred_id, ref_id]))
                   for reference, ref_id in NAME2ID.items()
                   for prediction, pred_id in NAME2ID.items()
                   if ref_id != pred_id and matrix[pred_id, ref_id] > 0]
    return dict(added_num=int(matrix[:-1, -1].sum()),
                lost_num=int(matrix[-1, :-1].sum()),
                category_changed_num=sum(item["count"] for item in transitions),
                category_transitions=transitions)


def compare(video: OpenCVVideoWrapper,
            base_dict: MDRF,
            new_dict: MDRF,
            pos_thre: float = 0.5,
            tiou: float = 0.3,
            aiou: float = 0.3,
            summary_out: dict | None = None) -> MDRF:
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

    # The temporal cursor below requires chronological targets. Export records
    # can arrive in completion order, including several targets in one record.
    base_results = sorted(get_regularized_results(base_dict, video),
                          key=lambda item: (item.start_frame, item.last_activate_frame))
    new_results = sorted(get_regularized_results(new_dict, video),
                         key=lambda item: (item.start_frame, item.last_activate_frame))

    mismatch_collection: list[MDTarget] = []

    # 主要指标
    # 统计时空匹配数，类别正确性由分类指标单独计算。
    gt_id = 0
    confusion_matrix = np.zeros((NUM_CLASS + 1, NUM_CLASS + 1), dtype=np.int64)

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
        while (gt_id < len(base_results)
               and instance.start_time >= base_results[gt_id].end_time):
            gt_id += 1

        # 为当前instance向后查找是否存在匹配
        match_flag = False
        cur_id = gt_id
        while (cur_id < len(base_results)
               and instance.end_time >= base_results[cur_id].start_time):
            if matched_id[cur_id] == 0 \
                and (calculate_time_iou(instance,base_results[cur_id]) >= tiou) \
                and calculate_area_iou(met2xyxy(instance.to_dict()), met2xyxy(base_results[cur_id].to_dict())) >= aiou:
                # TEMP FIX: 向前兼容v2.1.0的标注，低置信度转DROPPED进行判定。
                if gt_mode and base_results[cur_id].score <= pos_thre:
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
                matched_id[cur_id] = 1
                matched_pair_list.append((i, cur_id))
                break
            cur_id += 1
            if cur_id == len(base_results):
                match_flag = False
                break
        if not match_flag:
            confusion_matrix[NAME2ID[instance.category], -1] += 1

    # 完整记录未匹配参考目标，使 MISSED 行和丢失数一致。
    for index, reference in enumerate(base_results):
        if not matched_id[index]:
            reference_category = ("DROPPED" if gt_mode and reference.score <= pos_thre
                                  else reference.category)
            if reference_category == "UNKNOWN_AREA":
                reference_category = "OTHERS"
            confusion_matrix[-1, NAME2ID[reference_category]] += 1
    comparison_summary = summarize_comparison(confusion_matrix)
    if summary_out is not None:
        summary_out.update(comparison_summary)

    # 总候选数保留低分预测，用于观察候选数量变化。
    new_predict_num = len(new_results)
    old_predict_num = len(base_results)
    matched_num = int(np.count_nonzero(matched_id))
    #fn_list = np.array(base_results)[matched_id == 0]
    fn_num = old_predict_num - matched_num
    tn_num = new_predict_num - matched_num
    union_num = new_predict_num + old_predict_num - matched_num

    compare_result: dict[str, Union[int, float]] = {
        "matched_num":
        matched_num,
        "new_predict_num":
        new_predict_num,
        "low_score_predict_num":
        sum(instance.score <= pos_thre for instance in new_results),
        "old_predict_num":
        old_predict_num,
        "cross_ratio(A n B / A u B)":
        # 双方都为空时，约定结果完全一致。
        matched_num / union_num if union_num else 1.0,
        "fn_num":
        fn_num,
        "tn_num":
        tn_num,
        **comparison_summary
    }

    import pprint
    pprint.pprint(compare_result)
    print_confusion_matrix(confusion_matrix, list(NAME2ID.keys()) + ["MISSED"])

    return_dict = copy.deepcopy(new_dict)
    assert new_dict.anno_size is not None, "Invalid anno size..."
    return_dict.results = [
        SingleMDRecord.from_target(x, new_dict.anno_size)
        for x in mismatch_collection
    ]
    return return_dict



def generate_full_result(results: MDRF,
                         performance: dict[str, Union[float, str, None]]):
    # 补充必要信息
    assert isinstance(results.basic_info, BasicInfo), "Invalid basic info!"
    results.basic_info.desc = "待检测视频的基础信息 | Basic infomation about the video"
    performance["desc"] = "硬件指标 | Hardware performance"
    performance["cpu_core"] = psutil.cpu_count(logical=True)
    results.performance = performance
    return results


def load_cases(manifest_path, selected=()):
    """清单内的 json/cfg 路径相对于清单目录；case ID 必须唯一。"""
    manifest = Path(manifest_path).resolve()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("manifest requires a non-empty cases list")
    ids = set()
    resolved = []
    for case in cases:
        case = dict(case)
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in ids:
            raise ValueError("case IDs must be non-empty unique strings")
        ids.add(case_id)
        for key in ("json", "cfg"):
            if key == "json" or key in case:
                case[key] = str((manifest.parent / case[key]).resolve())
        resolved.append(case)
    unknown = set(selected) - ids
    if unknown:
        raise ValueError(f"unknown case IDs: {sorted(unknown)}")
    return [case for case in resolved if not selected or case["id"] in selected]


def run_batch(args):
    cases = (load_cases(args.manifest, args.case) if args.manifest else
             [dict(id=Path(args.report).stem, json=str(Path(args.report).resolve()))])
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(passes=args.passes, cases=[])
    for case_index, case in enumerate(cases, 1):
        case_summary = dict(id=case["id"], json=case["json"],
                            cfg=str(Path(case.get("cfg", args.cfg)).resolve()), runs=[])
        for pass_index in range(1, args.passes + 1):
            run_dir = output / f"case-{case_index:03d}" / f"pass-{pass_index:03d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            run_summary = run_dir / "run.json"
            command = [sys.executable, str(Path(__file__).resolve()), '--report', case["json"],
                       '--cfg', case_summary["cfg"], '--save-path', str(run_dir / "prediction.json"),
                       '--run-summary', str(run_summary)]
            if args.metric:
                command.extend(['--metric', '--metrics-path', str(run_dir / "metrics.json")])
            if args.debug:
                command.append('--debug')
            print(f"Case {case['id']} — pass {pass_index}/{args.passes}", flush=True)
            with open(run_dir / "run.log", "w", encoding="utf-8") as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode:
                record = dict(pass_index=pass_index, status="failed",
                              returncode=completed.returncode, log=str(run_dir / "run.log"))
            elif not run_summary.is_file():
                record = dict(pass_index=pass_index, status="failed",
                              error="evaluation did not produce a summary")
            else:
                record = json.loads(run_summary.read_text(encoding="utf-8"))
                record.update(pass_index=pass_index, status="ok", directory=str(run_dir))
            case_summary["runs"].append(record)
        successful = [run for run in case_summary["runs"] if run["status"] == "ok"]
        if successful:
            case_summary["performance_summary"] = {
                key: dict(median=float(np.median([run["performance"][key] for run in successful])),
                          min=min(run["performance"][key] for run in successful),
                          max=max(run["performance"][key] for run in successful))
                for key in ("tot_time", "cpu_time", "avg_cpu_usage", "peak_mem_usage")}
        summary["cases"].append(case_summary)
        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=4), encoding="utf-8")
    if any(run["status"] != "ok" for case in summary["cases"] for run in case["runs"]):
        raise SystemExit(1)


def main():
    # 可选模式
    # 1. 生成报告：对视频片段进行检测，给出当前版本下给定配置的报告。
    #    （视频片段的信息从给定的报告/GroundTruth中摘录得到）
    # 2. 效果回归：对当前视频与参考报告的检测效果和内存开销对比。
    #      a) 使用--load选项时，load选项作为当前的主结果。
    #      b) 当与annotation比较时，相当于计算检测指标；否则按照回归测试。
    # 3. TODO: 批处理：对一批数据执行类似操作。
    parser = argparse.ArgumentParser(description='MetDetPy Evaluater.')

    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--report', help="Single MDRF report JSON.")
    inputs.add_argument('--manifest', help="Evaluation manifest JSON containing a cases list.")

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
    parser.add_argument('--metrics-path', default=None,
                        help="Save classification metrics and PR points as JSON (requires --metric).")

    parser.add_argument('--case', action='append', default=[], help="Select a case ID; repeatable.")
    parser.add_argument('--passes', type=int, default=1, help="Runs per case (default: 1).")
    parser.add_argument('--output-dir', default='evaluation-results', help="Batch/repeat output directory.")
    parser.add_argument('--run-summary', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.passes < 1:
        parser.error('--passes must be positive')
    if args.case and not args.manifest:
        parser.error('--case requires --manifest')
    if args.manifest or args.passes > 1:
        if args.load:
            parser.error('--load is only supported for a single evaluation')
        if args.save_path or args.metrics_path:
            parser.error('batch/repeat uses --output-dir instead of --save-path/--metrics-path')
        run_batch(args)
        return
    if args.metrics_path and not args.metric:
        parser.error('--metrics-path requires --metric')

    ## Load video and config
    video_dict = MDRF.from_json_file(args.report)
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
    shared_path: str = os.path.split(args.report)[0]
    if os.path.split(video_name)[0] == "":
        video_name = os.path.join(shared_path, video_name)
        video_dict.basic_info.video = video_name
    if (mask_name) and (os.path.split(mask_name)[0] == ""):
        mask_name = os.path.join(shared_path, mask_name)
        video_dict.basic_info.mask = mask_name

    metrics = None
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
            metrics = calculate_detection_metrics(
                get_regularized_results(video_dict, video),
                get_regularized_results(new_result, video),
                gt_mode=video_dict.type == "annotation")
            import pprint
            pprint.pprint(metrics["metrics"])
            comparison_summary = {}
            mismatch = compare(video,
                               base_dict=video_dict,
                               new_dict=new_result,
                               summary_out=comparison_summary)
            metrics["comparison_summary"] = comparison_summary
            if args.metrics_path:
                with open(args.metrics_path, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, ensure_ascii=False, indent=4)
            mismatch_path = (str(Path(args.run_summary).with_name("mismatch.json"))
                             if args.run_summary else "mismatch.json")
            with open(mismatch_path, mode="w", encoding="utf-8") as f:
                json.dump(mismatch.to_dict(), f, ensure_ascii=False, indent=4)
        if args.run_summary:
            with open(args.run_summary, "w", encoding="utf-8") as f:
                json.dump(dict(performance=new_result.performance, metrics=metrics),
                          f, ensure_ascii=False, indent=4)
    finally:
        video.release()


if __name__ == "__main__":
    main()
