"""

ClipToolkit 可用于一次性创建一个视频中的多段视频切片或视频段的堆栈图像。

从v2.2.0开始，扩增了支持的输入风格，以支持更灵活的使用和更通用的场景。支持了以下主要调用方法：

1. 同时提供target视频与复数个片段的json。（延续v1.3.0开始的风格。）
    示例：python evaluate.py "test/20220413_annotation.json"

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
import cv2

from MetLib.Stacker import max_stacker, all_stacker
from MetLib.utils import frame2ts, save_img, save_video, ts2frame
from MetLib.VideoLoader import ThreadVideoLoader
from MetLib.VideoWrapper import OpenCVVideoWrapper
from MetLib.MetLog import get_default_logger, set_default_logger

support_image_suffix = ["JPG", "JPEG", "PNG"]
support_video_suffix = ["AVI"]


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
    argparser.add_argument(
        "--start-time",
        type=str,
        help=
        "start time of the video. Optional. Support int in ms or format like \"HH:MM:SS\". "
        "If not provided, it will start at 0 frame as default")
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

    argparser.add_argument("--resize",
                           type=str,
                           help="resize img/video to the given solution.",
                           default=None)

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

    argparser.add_argument("--debug",
                           action="store_true",
                           help="apply debug mode.")

    args = argparser.parse_args()

    # basic option
    target_name, json_str, mode, default_suffix, resize, save_path, debug_mode  = \
        args.target, args.json, args.mode, args.suffix, args.resize, args.save_path, args.debug

    # image option
    jpg_quality, png_compress = args.jpg_quality, args.png_compressing

    if json_str is not None:
        # 提供json_str时，按照旧版本逻辑执行
        video_name = target_name
        # parse json argument
        data = None
        if os.path.isfile(json_str):
            with open(json_str, mode='r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = json.loads(json_str)
    elif target_name.split(".")[-1].lower() == "json":
        # 仅提供json时，按照判断符合格式，具有足够信息
        if os.path.isfile(target_name):
            with open(target_name, mode='r', encoding='utf-8') as f:
                raw_data: dict = json.load(f)
        else:
            raise FileNotFoundError(
                f"{target_name} can not be opened as a file.")
        if not (raw_data.get("basic_info", None)
                or raw_data.get("results", None)):
            raise ValueError(
                f"{target_name} is not a valid json file for ClipToolkit.")
        video_name = raw_data["basic_info"]["video"]
        data = raw_data["results"]
    else:
        # target被作为视频解析。从参数构造单个使用的data。
        video_name = target_name
        data = [dict(start_time=args.start_time, end_time=args.end_time)]

    video_loader = ThreadVideoLoader(OpenCVVideoWrapper,
                                     video_name,
                                     resize_option=resize,
                                     exp_option="real-time",
                                     resize_interpolation=cv2.INTER_LANCZOS4)

    # get video name
    _, video_name_nopath = os.path.split(video_name)
    video_name_pure = ".".join(video_name_nopath.split(".")[:-1])

    # get default suffix
    if default_suffix is None:
        if mode == "image": default_suffix = "jpg"
        if mode == "video": default_suffix = "avi"

    # 单一片段时，若保存路径中包含文件名，覆盖输出的文件名。
    if len(data) == 1:
        if not os.path.isdir(save_path):
            save_path, filename = os.path.split(save_path)
            data[0]["filename"] = filename

    # 获取Logger
    logger = get_default_logger()
    set_default_logger(debug_mode, work_mode="frontend")
    try:
        logger.start()
        for single_data in data:
            if "time" in single_data:
                start_time, end_time = single_data["time"]
            else:
                start_time, end_time = single_data["start_time"], single_data[
                    "end_time"]
            # 如果未给定起止时间，使用视频的起止时间
            if start_time is None:
                start_time = frame2ts(video_loader.start_frame,
                                      video_loader.fps)
            if end_time is None:
                end_time = frame2ts(video_loader.end_frame, video_loader.fps)
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
                logger.error(
                    f"Unsupport suffix: {suffix}. Ignore error and continue.")
                continue

            full_path = os.path.join(save_path, tgt_name)
            video_loader.reset(ts2frame(start_time, video_loader.fps),
                               ts2frame(end_time, video_loader.fps))

            if cur_mode == "image":
                results = max_stacker(video_loader)
                if results is not None:
                    save_img(results, full_path, jpg_quality, png_compress)
                    logger.info(f"Saved: {full_path}")
                else:
                    logger.error("Error occured, got empty image.")
            else:
                video_series = all_stacker(video_loader)
                if video_series is not []:
                    save_video(video_series, video_loader.fps, full_path)
                    logger.info(f"Saved: {full_path}")
                else:
                    logger.error("Error occured, got empty video clip.")
    finally:
        logger.stop()


if __name__ == "__main__":
    main()
