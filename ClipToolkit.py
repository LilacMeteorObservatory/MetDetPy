import argparse
import json
import os
import cv2

from MetLib.Stacker import max_stacker, all_stacker
from MetLib.utils import save_img, save_video, time2frame
from MetLib.VideoLoader import ThreadVideoReader
from MetLib.VideoWarpper import OpenCVVideoWarpper
from MetLib.MetLog import get_default_logger

support_image_suffix = ["JPG", "JPEG", "PNG"]
support_video_suffix = ["AVI"]


def main():
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

    args = argparser.parse_args()

    # basic option
    video_name, json_str, mode, default_suffix, resize, save_path,  = \
        args.target, args.json, args.mode, args.suffix, args.resize, args.save_path,

    # image option
    jpg_quality, png_compress = args.jpg_quality, args.png_compressing

    video_loader = ThreadVideoReader(OpenCVVideoWarpper,
                                     video_name,
                                     resize_option=resize,
                                     exp_option=1,
                                     resize_interpolation=cv2.INTER_LANCZOS4)

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

    # 单一片段时，若保存路径中包含文件名，覆盖输出的文件名。
    if len(data) == 1:
        if not os.path.isdir(save_path):
            save_path, filename = os.path.split(save_path)
            data[0]["filename"] = filename
        data = [data]

    # 获取Logger
    logger = get_default_logger()

    try:
        logger.start()
        for single_data in data:
            if "time" in single_data:
                start_time, end_time = single_data["time"]
            else:
                start_time, end_time = single_data["start_time"], single_data[
                    "end_time"]
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
                print(
                    f"Unsupport suffix: {suffix}. Ignore error and continue.")
                continue

            full_path = os.path.join(save_path, tgt_name)
            video_loader.reset(time2frame(start_time, video_loader.fps),
                               time2frame(end_time, video_loader.fps))

            if cur_mode == "image":
                results = max_stacker(video_loader)
                save_img(results, full_path, jpg_quality, png_compress)
            else:
                video_series = all_stacker(video_loader)
                save_video(video_series, video_loader.fps, full_path)
            logger.info(f"Saved: {full_path}")

    finally:
        logger.stop()


if __name__ == "__main__":
    main()