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
[--mode {image,video}] [--suffix SUFFIX] [--save-path SAVE_PATH] [--resize RESIZE] [--jpg-quality JPG_QUALITY] [--png-compressing PNG_COMPRESSING]

"""

import argparse
import json
import os
import time
from os.path import join as path_join
from os.path import split as path_split
from typing import Any, Optional, cast

import cv2
from dacite import from_dict

from MetLib import *
from MetLib.fileio import (change_file_path, is_ext_with, load_8bit_image,
                           replace_path_ext, save_img)
from MetLib.metlog import BaseMetLog, get_default_logger, set_default_logger
from MetLib.metstruct import (MDRF, ClipCfg, ClipRequest, ExportOption,
                              ImageFrameData, SimpleTarget, VideoFrameData)
from MetLib.stacker import (max_stacker, mfnr_mix_stacker,
                            simple_denoise_stacker)
from MetLib.utils import CLIP_CONFIG_PATH, U8Mat, frame2ts, ts2frame

support_image_suffix = ["JPG", "JPEG", "PNG"]
support_video_suffix = ["AVI"]
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


def draw_target(img: U8Mat, target_list: Optional[list[SimpleTarget]],
                cfg: ExportOption):
    if target_list is None:
        return img
    for target in target_list:
        img = cv2.rectangle(img,
                            target.pt1,
                            target.pt2,
                            color=cfg.bbox_color,
                            thickness=cfg.bbox_thickness)
    return img


def jsonsf2request(json_str: str):
    data: Optional[list[dict[str, Any]]] = None
    if os.path.isfile(json_str):
        with open(json_str, mode='r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = json.loads(json_str)
    assert isinstance(data, list), "Json must be a list!"
    return [
        from_dict(data_class=ClipRequest, data=req).to_video_data()
        for req in data
    ]


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

        mdrf_data = from_dict(data_class=MDRF, data=raw_data)
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


def image_clip_process(data: list[ImageFrameData], export_cfg: ExportOption,
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
        for frame_data in data:
            image_data = load_8bit_image(frame_data.img_filename)
            if image_data is None:
                logger.warning(
                    f"Failed to load {frame_data.img_filename}, skip...")
                continue
            # 填充video_size: 转换image标注为对应格式
            frame_data.img_size = image_data.shape[:2][1::-1]
            if export_cfg.with_bbox:
                image_data = draw_target(image_data, frame_data.target_list,
                                         export_cfg)
            # 保存图像到目标路径下
            full_path = change_file_path(frame_data.img_filename, save_path)
            save_img(image_data,
                     full_path,
                     export_cfg.jpg_quality,
                     export_cfg.png_compressing,
                     color_space='sRGB',
                     logger=logger)
            logger.info(f"Saved: {full_path}")
            # 在有target的情况下，同时生成labelme风格的标注
            if export_cfg.with_annotation:
                res_dict = frame_data.to_labelme()
                if res_dict:
                    anno_path = replace_path_ext(full_path, ".json")
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

    argparser.add_argument("--debayer",
                           action="store_true",
                           help="apply debayer for video mode.")
    argparser.add_argument("--debayer-pattern",
                           help="debayer pattern, like RGGB or BGGR.")

    argparser.add_argument("--debug",
                           action="store_true",
                           help="apply debug mode.")

    args = argparser.parse_args()
    
    t0 = time.time()
    # basic option
    cfg_json_path, mode, default_suffix, save_path, debug_mode  = \
        args.cfg, args.mode, args.suffix, args.save_path, args.debug

    # image option
    jpg_quality, png_compress = args.jpg_quality, args.png_compressing

    # 获取，同步从命令行获取的配置
    with open(cfg_json_path, mode='r', encoding='utf-8') as f:
        cfg_json = json.load(f)
    clip_cfg = from_dict(data_class=ClipCfg, data=cfg_json)
    update_cfg_from_args(clip_cfg, args)

    denoise_cfg = clip_cfg.image_denoise
    export_cfg = clip_cfg.export

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
                           clip_cfg.export,
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
                                  exp_option="real-time",
                                  resize_interpolation=cv2.INTER_LANCZOS4,
                                  debayer=args.debayer,
                                  debayer_pattern=args.debayer_pattern)
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
                    logger.error(
                        f"Fail to generate image for data: {video_loader.video_name}"
                        f" with start-time={video_loader.start_time} "
                        f"and end-time={video_loader.end_time}.")
                    continue
                if export_cfg.with_bbox:
                    results = draw_target(results, video_frame.target_list,
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
                    res_dict = video_frame.to_labelme()
                    anno_path = replace_path_ext(video_frame.saved_filename,
                                                 "json")
                    with open(anno_path, mode="w", encoding="utf-8") as f:
                        json.dump(res_dict, f, ensure_ascii=False, indent=4)
                    logger.info(f"Saved: {anno_path}")
            else:
                status_code = VideoWriterCls.save_video_by_stream(
                    video_loader,
                    video_loader.fps,
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
