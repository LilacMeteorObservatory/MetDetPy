"""

ClipToolkit 可用于一次性创建一个视频中的多段视频切片或视频段的堆栈图像。

从v2.2.0开始，扩增了支持的输入风格，以支持更灵活的使用和更通用的场景。支持了以下主要调用方法：

1. 同时提供target视频与复数个片段的json。（延续v1.3.0开始的风格。）
    示例：python ClipToolkit.py target "test/20220413_annotation.json"

2. 当仅处理单张图像时，可以仅指定target视频，并在 optional args中使用简化的输入接口：
    示例：python ClipToolkit.py target --start-time 00:03:00 --end-time 00:05:00 --mode image --output-name 123.jpg

3. 当处理检测结果（或标注）时，可以仅指定v2.2.0之后的evaluate或MetDetPy生成的json作为输入。
    示例：python ClipToolkit.py execution_result.json --mode video

可选参数：
[--mode {image,video}] [--suffix SUFFIX] [--save-path SAVE_PATH] [--resize RESIZE] [--jpg-quality JPG_QUALITY] [--png-compressing PNG_COMPRESSING] [--padding-before PADDING_BEFORE] [--padding-after PADDING_AFTER]

"""

import argparse
import json
import os
import shutil
import time
from os.path import join as path_join
from os.path import split as path_split
from typing import Any, Optional, cast

import cv2

from MetLib import *
from MetLib.fileio import (SUPPORT_RAW_FORMAT, change_file_path, is_ext_with,
                           is_ext_within, load_image_file, replace_path_ext,
                           save_img)
from MetLib.metlog import BaseMetLog, get_default_logger, set_default_logger
from MetLib.metstruct import (MDRF, BasicInfo, ClipCfg, ClipRequest,
                              ExportOption, FilterRules, ImageFrameData,
                              SimpleTarget, VideoFrameData)
from MetLib.stacker import (all_stacker, max_stacker, mfnr_mix_stacker,
                            simple_denoise_stacker)
from MetLib.utils import (CLIP_CONFIG_PATH, U8Mat, adjust_ts, frame2ts, pt_len,
                           set_resource_dir, ts2frame)

support_image_suffix = ["JPG", "JPEG", "PNG"]
support_video_suffix = ["AVI", "MP4"]
IMAGE_MODE = "image"
VIDEO_MODE = "video"
DEFAULT_SUFFIX_MAPPING = {IMAGE_MODE: "jpg", VIDEO_MODE: "avi"}
NO_VIDEO_PROMPT = "Missed video name in input MDRF files. Check `video` in `basic_info` part."
MFNR = "mfnr-mix"
SDS = "simple"
AVAILABLE_STACKER_MAPPING = {
    MFNR: mfnr_mix_stacker,
    SDS: simple_denoise_stacker
}
BUILTIN_NEGATIVE_CATEGORIES = {"DROPPED", "OTHERS"}


def adaptive_font_param(img: U8Mat) -> dict[str, int]:
    short_length = min(img.shape[0], img.shape[1])
    return {
        "font_offset": round(short_length / 2000) + 4,
        "font_scale": round(short_length / 2000),
        "font_thickness": int(max(1, short_length // 750))
    }


def update_cfg_from_args(base_cfg: ClipCfg, args: argparse.Namespace):
    """Sync args modification to base_cfg.

    Args:
        base_cfg (ClipCfg): Config from the file.
        args (argparse.Namespace): command line args.
    """
    base_cfg.image_denoise.switch = args.denoise is not None
    base_cfg.image_denoise.algorithm = args.denoise
    base_cfg.export.jpg_quality = args.jpg_quality
    base_cfg.export.png_compressing = args.png_compressing
    base_cfg.export.with_bbox = args.with_bbox
    base_cfg.export.with_annotation = args.with_annotation
    if args.padding_before is not None:
        base_cfg.export.clip_padding.before = args.padding_before
    if args.padding_after is not None:
        base_cfg.export.clip_padding.after = args.padding_after
    if args.filter_rules_switch is not None:
        base_cfg.export.filter_rules.switch = args.filter_rules_switch


def draw_target(img: U8Mat, target_list: Optional[list[SimpleTarget]],
                cfg: ExportOption) -> U8Mat:
    """draw positive target on the image.

    Args:
        img (U8Mat): base image
        target_list (Optional[list[SimpleTarget]]): target list
        cfg (ExportOption): Export Option

    Returns:
        U8Mat: image with annotations
    """
    if target_list is None:
        return img
    for target in target_list:
        color = cfg.bbox_color
        if cfg.bbox_color_mapping and target.preds in cfg.bbox_color_mapping:
            color = cfg.bbox_color_mapping[target.preds]
        img = cv2.rectangle(img,
                            target.pt1,
                            target.pt2,
                            color=color,
                            thickness=cfg.bbox_thickness)
        # 在bbox上方绘制类别文字和概率，使用 getTextSize 精确测量并处理越界
        font_param = adaptive_font_param(img)
        text = f"{target.preds}: {target.prob}"
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fs = font_param["font_scale"]
        th = font_param["font_thickness"]
        offset = font_param["font_offset"]

        # 测量文本的像素尺寸（宽，高）
        (_, text_h), _ = cv2.getTextSize(text, fontFace, fs, th)

        # 先尝试放在 bbox 上方：baseline 放在 bbox.top - offset
        proposed_baseline = int(target.pt1[1] - offset)
        if proposed_baseline - text_h < 0:
            # 放在 bbox 下方，baseline 放在 bbox.bottom + offset + text_h
            baseline_pos = int(target.pt2[1] + offset + text_h)
        else:
            baseline_pos = proposed_baseline

        # 确保 baseline 不会超过图像底部
        max_baseline = img.shape[0] - 1
        if baseline_pos > max_baseline:
            baseline_pos = max(max_baseline, text_h)

        img = cv2.putText(img,
                          text, (int(target.pt1[0]), int(baseline_pos)),
                          fontFace=fontFace,
                          fontScale=fs,
                          color=color,
                          thickness=th)
    return img


def jsonsf2request(json_str: str):
    """convert json_str in argument to be a list.
    
    Args:
        json_str (str): source json string, could be a json string or a path to a json file.

    Returns:
        list[VideoFrameData]: parsed json list.
    """
    data: Optional[list[dict[str, Any]]] = None
    if os.path.isfile(json_str):
        with open(json_str, mode='r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = json.loads(json_str)
    assert isinstance(data, list), "Json must be a list!"
    return [ClipRequest.from_dict(req).to_video_data() for req in data]


def parse_input(target_name: str, json_str: Optional[str], logger: BaseMetLog,
                args: Any):
    """根据命令行输入解析，拆分任务形态。

    Args:
        target_name (str): _description_
        json_str (Optional[str]): _description_

    Raises:
        FileNotFoundError: _description_
    """
    if json_str is not None:
        # 提供 target_name 和 json_str时，按照旧版本逻辑，将json转换为 list[VideoFrameData].
        video_name = target_name
        request_list = jsonsf2request(json_str)
        return video_name, request_list
    elif is_ext_with(target_name, "json"):
        # 仅提供json（作为target_name）时，尝试按照MDRF解析
        if os.path.isfile(target_name):
            with open(target_name, mode='r', encoding='utf-8') as f:
                raw_data: dict[str, Any] = json.load(f)
        else:
            raise FileNotFoundError(
                f"{target_name} can not be opened as a file.")

        mdrf_data = MDRF.from_dict(raw_data)
        video_name = mdrf_data.basic_info.video
        data = mdrf_data.results
        # 根据 MDRF 类型分流处理
        if mdrf_data.type in ("image-prediction", "timelapse-prediction"):
            # 转换图像检测结果格式
            if len(data) == 0:
                logger.warning("Empty result is provided.")
            if raw_data["type"] == "image-prediction":
                frame_data_list = [
                    single_record.to_image_data() for single_record in data
                ]
                return None, frame_data_list
            else:
                assert isinstance(mdrf_data.basic_info,
                                  BasicInfo), "Invalid MDRF basic_info type."
                video_data_list = [
                    single_record.to_video_data(fps=mdrf_data.basic_info.fps,
                                                video_size=mdrf_data.anno_size)
                    for single_record in data
                ]
                assert video_name is not None, NO_VIDEO_PROMPT
                return video_name, video_data_list
        else:
            # 视频检测结果数据，直接转换
            assert video_name is not None, NO_VIDEO_PROMPT
            video_data_list = [
                single_record.to_video_data() for single_record in data
            ]
            return video_name, video_data_list
    else:
        # target 直接被作为视频解析。从参数构造单个使用的data。
        request_list = [
            VideoFrameData(start_time=args.start_time,
                           end_time=args.end_time,
                           target_list=None,
                           video_size=None)
        ]
        return target_name, request_list



def filter_targets(target_list: Optional[list[SimpleTarget]],
                   filter_rules: FilterRules,
                   diag_length: int) -> list[SimpleTarget]:
    """Filter targets by export rules and return retained targets."""
    if target_list is None:
        return []
    if diag_length <= 0:
        return []
    retained_targets: list[SimpleTarget] = []
    for target in target_list:
        # Always exclude built-in negative classes once filtering is enabled.
        if target.preds in BUILTIN_NEGATIVE_CATEGORIES:
            continue
        if target.preds in filter_rules.exclude_category_list:
            continue
        if target.prob is None or float(target.prob) < filter_rules.threshold:
            continue
        if pt_len(target.pt1,
                  target.pt2) / diag_length < filter_rules.min_length_ratio:
            continue
        retained_targets.append(target)
    return retained_targets


def image_clip_process(data: list[ImageFrameData], clip_cfg: ClipCfg,
                       save_path: str, logger: BaseMetLog):
    """ 图像序列->筛选图像序列的保存接口。

    Args:
        data (list[ImageFrameData]): _description_
        clip_cfg (ClipCfg): _description_
        save_path (str): _description_
        logger (BaseMetLog): _description_
    """
    try:
        logger.start()
        export_cfg, raw_cfg = clip_cfg.export, clip_cfg.raw_img_load_config
        filter_rules = export_cfg.filter_rules
        for frame_data in data:
            image_data = None
            diag_length = 0
            target_list = frame_data.target_list
            # NOTE: 兼容旧输出格式的设计（缺少img_size字段）
            # 新MDRF 会携带 img_size 字段，无需加载图像; 旧版本则需要预先加载图像。
            # v3.0.0后放弃对旧输出格式的兼容。
            if frame_data.img_size is not None:
                diag_length = pt_len([0, 0], list(frame_data.img_size))
            else:
                image_data = load_image_file(frame_data.img_filename, raw_cfg,
                                             logger)
                if image_data is None:
                    continue
                diag_length = pt_len([0, 0], list(image_data.shape[:2]))
                frame_data.img_size = image_data.shape[:2][1::-1]

            if filter_rules.switch:
                target_list = filter_targets(frame_data.target_list,
                                             filter_rules, diag_length)
            # 如果所有target都被过滤规则过滤，则跳过
            if filter_rules.switch and not target_list:
                logger.info(
                    f"Skip {frame_data.img_filename} because no valid target in this image."
                )
                continue

            full_path = change_file_path(frame_data.img_filename, save_path)
            if export_cfg.with_bbox:
                # 仅在需要导出 bbox 时，输入图像被载入。
                if image_data is None:
                    image_data = load_image_file(frame_data.img_filename,
                                                 raw_cfg, logger)
                    if image_data is None:
                        continue
                image_data = draw_target(image_data, target_list, export_cfg)
                # 保存图像到目标路径下
                if is_ext_within(full_path, SUPPORT_RAW_FORMAT):
                    logger.warning(
                        f"Cannot draw targets on .{frame_data.img_filename} format image"
                        ", save .jpg instead.")
                    full_path = replace_path_ext(full_path, 'jpg')
                save_img(image_data,
                         full_path,
                         export_cfg.jpg_quality,
                         export_cfg.png_compressing,
                         color_space='sRGB',
                         logger=logger)
                logger.info(f"Saved: {full_path}")
            else:
                # 不绘制bbox，则直接copy原始文件。
                shutil.copy(frame_data.img_filename, full_path)
                logger.info(f"Copied: {full_path}")
            # 在有target的情况下，同时生成labelme风格的标注
            if export_cfg.with_annotation:
                frame_data.target_list = target_list
                res_dict = frame_data.to_labelme()
                if res_dict:
                    anno_path = replace_path_ext(full_path, "json")
                    with open(anno_path, mode="w", encoding="utf-8") as f:
                        json.dump(res_dict, f, ensure_ascii=False, indent=4)
                    logger.info(f"Saved: {anno_path}")
    except Exception as e:
        logger.error(
            f"Fatal error occured: {e.__repr__()}. Process is interrupted.")
    finally:
        logger.stop()
    return


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("target", type=str, help="the target video.")
    argparser.add_argument(
        "json",
        type=str,
        nargs='?',
        default=None,
        help=
        "a json-format string or the path to a json file where start-time and end-time are listed."
    )
    argparser.add_argument("--cfg",
                           "-C",
                           type=str,
                           help="Path to the config file.",
                           default=CLIP_CONFIG_PATH)
    argparser.add_argument(
        "--start-time",
        type=str,
        help=
        "start time of the video. Optional. Support int in ms or format like \"HH:MM:SS\". "
        "If not provided, it will start at 0 frame as default.")
    argparser.add_argument(
        "--end-time",
        type=str,
        help=
        "end time of the clip. Optional. Support int in ms or format like \"HH:MM:SS\". "
        "If not provided, it will use END_TIME as default.")
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
        help=
        "the path where image(s)/video(s) are placed. When only one clip is provided,\
            included filename will be used as filename.",
        default=os.getcwd())

    img_group_args = argparser.add_argument_group(
        "optional image-related arguments")

    img_group_args.add_argument(
        "--png-compressing",
        type=int,
        help=
        "the compressing of generated png image. It should be int ranged Z in [0,9];\
            By default, it is 3.",
        default=3)

    img_group_args.add_argument(
        "--jpg-quality",
        type=int,
        help=
        "the quality of generated jpg image. It should be int ranged Z in [0,100];\
            By default, it is 95.",
        default=95)
    img_group_args.add_argument("--denoise",
                                type=str,
                                choices=AVAILABLE_STACKER_MAPPING.keys(),
                                help="optional denoise algorithm. ",
                                default=None)

    argparser.add_argument("--with-annotation",
                           action="store_true",
                           help="generate labelme style annotation.")

    argparser.add_argument("--with-bbox",
                           action="store_true",
                           help="draw bounding box contours with red line.")
    filter_rule_group = argparser.add_mutually_exclusive_group()
    filter_rule_group.add_argument(
        "--enable-filter-rules",
        dest="filter_rules_switch",
        action="store_true",
        help=
        "enable export.filter_rules.switch from command line and override config."
    )
    filter_rule_group.add_argument(
        "--disable-filter-rules",
        dest="filter_rules_switch",
        action="store_false",
        help=
        "disable export.filter_rules.switch from command line and override config."
    )
    argparser.set_defaults(filter_rules_switch=None)

    argparser.add_argument("--debayer",
                           action="store_true",
                           help="apply debayer for video mode.")
    argparser.add_argument("--debayer-pattern",
                           help="debayer pattern, like RGGB or BGGR.")

    argparser.add_argument("--debug",
                           action="store_true",
                           help="apply debug mode.")
    argparser.add_argument("--resource-dir", "-R",
                           type=str,
                           help="Path to the resource folder (config/weights/resource/global).",
                           default=None)

    argparser.add_argument("--padding-before",
                           type=float,
                           help="padding time before the clip start (in seconds). "
                           "Overrides the config file setting.",
                           default=None)
    argparser.add_argument("--padding-after",
                           type=float,
                           help="padding time after the clip end (in seconds). "
                           "Overrides the config file setting.",
                           default=None)

    args = argparser.parse_args()
    
    if args.resource_dir:
        set_resource_dir(args.resource_dir)


    t0 = time.time()
    # basic option
    cfg_json_path, mode, default_suffix, save_path, debug_mode  = \
        args.cfg, args.mode, args.suffix, args.save_path, args.debug

    # image option
    jpg_quality, png_compress = args.jpg_quality, args.png_compressing

    # 获取，同步从命令行获取的配置
    with open(cfg_json_path, mode='r', encoding='utf-8') as f:
        cfg_json = json.load(f)
    clip_cfg = ClipCfg.from_dict(cfg_json)
    update_cfg_from_args(clip_cfg, args)

    denoise_cfg = clip_cfg.image_denoise
    export_cfg = clip_cfg.export
    filter_rules = export_cfg.filter_rules

    # 获取Logger
    logger = get_default_logger()
    set_default_logger(debug_mode, work_mode="frontend")

    video_name, request_list = parse_input(args.target,
                                           args.json,
                                           logger=logger,
                                           args=args)

    # save_path valid check
    if len(request_list) == 1 and request_list[0].saved_filename is None:
        # 只有一个视频时候，通过output如果指定名称(有后缀名)，直接作为最终输出
        if os.path.splitext(save_path)[-1]:
            save_path, request_list[0].saved_filename = path_split(save_path)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if video_name is None:
        # Image Folder Mode, early return
        request_list = cast(list[ImageFrameData], request_list)
        image_clip_process(request_list,
                           clip_cfg,
                           save_path=save_path,
                           logger=logger)
        logger.stop()
        return

    request_list = cast(list[VideoFrameData], request_list)
    VideoLoaderCls = get_loader(clip_cfg.loader)
    VideoWrapperCls = get_wrapper(clip_cfg.wrapper)
    video_loader = VideoLoaderCls(VideoWrapperCls,
                                  video_name,
                                  resize_option=None,
                                  hwaccel=None,
                                  exp_option="real-time",
                                  resize_interpolation=cv2.INTER_LANCZOS4,
                                  debayer=args.debayer,
                                  debayer_pattern=args.debayer_pattern,
                                  continue_on_err=True)
    VideoWriterCls = get_writer(clip_cfg.writer)
    # get video name
    _, video_name_nopath = path_split(video_name)
    video_name_pure = os.path.splitext(video_name_nopath)[0]
    # get default suffix
    if default_suffix is None:
        default_suffix = DEFAULT_SUFFIX_MAPPING.get(mode, default_suffix)

    # 单一片段时，若保存路径中包含文件名，覆盖输出的文件名。
    if len(request_list) == 1:
        if not os.path.isdir(save_path):
            save_path, filename = path_split(save_path)
            request_list[0].saved_filename = filename

    try:
        logger.start()
        for video_frame in request_list:
            # 如果未给定起止时间，使用视频的起止时间
            if video_frame.start_time is None:
                video_frame.start_time = frame2ts(video_loader.start_frame,
                                                  video_loader.fps)
            if video_frame.end_time is None:
                video_frame.end_time = frame2ts(video_loader.end_frame,
                                                video_loader.fps)
            # 应用 clip_padding 补偿
            if export_cfg.clip_padding.before != 0.0:
                video_frame.start_time = adjust_ts(
                    video_frame.start_time,
                    -export_cfg.clip_padding.before,
                    video_loader.fps)
            if export_cfg.clip_padding.after != 0.0:
                video_frame.end_time = adjust_ts(
                    video_frame.end_time,
                    export_cfg.clip_padding.after,
                    video_loader.fps)

            # 边界检查：确保时间戳在视频有效范围内
            start_frame = ts2frame(video_frame.start_time, video_loader.fps)
            end_frame = ts2frame(video_frame.end_time, video_loader.fps)
            if start_frame < 0:
                logger.warning(
                    f"Clip start_time {video_frame.start_time} (frame {start_frame}) "
                    f"is before video start. Clipping to video start.")
                video_frame.start_time = frame2ts(0,
                                                  video_loader.fps)

            if end_frame > video_loader.video_total_frames:
                logger.warning(
                    f"Clip end_time {video_frame.end_time} (frame {end_frame}) "
                    f"is after video end. Clipping to video end.")
                video_frame.end_time = frame2ts(video_loader.video_total_frames,
                                                video_loader.fps)

            # 如果未给定名称则使用缺省名称
            tgt_name = video_frame.saved_filename if video_frame.saved_filename else f"{video_name_pure}_{video_frame.start_time}-{video_frame.end_time}.{default_suffix}"
            tgt_name = tgt_name.replace(":", "_")

            # 获取后缀，检查后缀合法性
            cur_mode = mode
            suffix = os.path.splitext(tgt_name)[-1].replace(".", "").upper()
            if suffix in support_image_suffix: cur_mode = IMAGE_MODE
            elif suffix in support_video_suffix: cur_mode = VIDEO_MODE
            else:
                logger.error(
                    f"Unsupport suffix: {suffix}. Ignore error and continue.")
                continue

            video_frame.saved_filename = path_join(save_path, tgt_name)
            video_loader.reset(
                ts2frame(video_frame.start_time, video_loader.fps),
                ts2frame(video_frame.end_time, video_loader.fps))
            frame_target_list = video_frame.target_list
            if filter_rules.switch:
                diag_length = 0
                if video_frame.video_size is not None:
                    diag_length = pt_len([0, 0], list(video_frame.video_size))
                else:
                    diag_length = pt_len([0, 0], video_loader.raw_size)
                frame_target_list = filter_targets(video_frame.target_list,
                                                   filter_rules, diag_length)
                if not frame_target_list:
                    logger.debug(
                        f"Skip {video_frame.saved_filename} because no valid target in this clip."
                    )
                    continue

            if cur_mode == IMAGE_MODE:
                results = None
                if denoise_cfg.switch:
                    assert denoise_cfg.algorithm in AVAILABLE_STACKER_MAPPING, "unsupport denoise algorithm!"
                    denoise_stacker = AVAILABLE_STACKER_MAPPING[
                        denoise_cfg.algorithm]
                    results = denoise_stacker(video_loader,
                                              denoise_cfg,
                                              logger=logger)
                else:
                    results = max_stacker(video_loader)
                if results is None:
                    logger.fatal(
                        f"Failed to generate image for data: {video_loader.video_name}"
                        f" with start-time={video_loader.start_time} "
                        f"and end-time={video_loader.end_time}.")
                    continue
                if export_cfg.with_bbox:
                    results = draw_target(results, frame_target_list,
                                          clip_cfg.export)
                # img save
                if results is not None:
                    save_img(results,
                             video_frame.saved_filename,
                             jpg_quality,
                             png_compress,
                             color_space='sRGB',
                             logger=logger)
                    logger.info(f"Saved: {video_frame.saved_filename}")
                else:
                    logger.error("Error occured, got empty image.")
                # 在有target的情况下，同时生成labelme风格的标注
                if export_cfg.with_annotation:
                    video_frame.target_list = frame_target_list
                    res_dict = video_frame.to_labelme()
                    anno_path = replace_path_ext(video_frame.saved_filename,
                                                 "json")
                    with open(anno_path, mode="w", encoding="utf-8") as f:
                        json.dump(res_dict, f, ensure_ascii=False, indent=4)
                    logger.info(f"Saved: {anno_path}")
            else:
                if export_cfg.with_bbox:
                    img_series = all_stacker(video_loader, logger=logger)
                    if img_series is not None:
                        post_img_series = [
                            draw_target(img, frame_target_list,
                                        clip_cfg.export) for img in img_series
                        ]
                        status_code = VideoWriterCls.save_video_with_audio(
                            post_img_series,
                            video_loader,
                            clip_cfg.export,
                            video_frame.saved_filename,
                            start_frame=video_loader.start_frame,
                            end_frame=video_loader.end_frame,
                            logger=logger)
                    else:
                        status_code = -1
                else:
                    status_code = VideoWriterCls.save_video_by_stream(
                        video_loader,
                        clip_cfg.export,
                        video_frame.saved_filename,
                        logger=logger)
                if status_code == 0:
                    logger.info(f"Saved: {video_frame.saved_filename}")
                else:
                    logger.error(
                        f"Error occured when writing the video to {video_frame.saved_filename}."
                    )
    finally:
        logger.debug(f"Time cost: {(time.time()-t0):.2f}s.")
        logger.stop()


if __name__ == "__main__":
    main()
