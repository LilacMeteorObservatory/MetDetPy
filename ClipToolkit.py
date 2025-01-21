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

import cv2

from MetLib.MetLog import get_default_logger, set_default_logger
from MetLib.Stacker import max_stacker
from MetLib.utils import (frame2ts, list2xyxy, save_img, save_video_by_stream,
                          ts2frame, load_8bit_image)
from MetLib.VideoLoader import ThreadVideoLoader
from MetLib.VideoWrapper import OpenCVVideoWrapper

support_image_suffix = ["JPG", "JPEG", "PNG"]
support_video_suffix = ["AVI"]


def generate_labelme(single_data: dict, img_fn: str) -> dict:
    if not "target" in single_data:
        return {}
    w, h = single_data["video_size"]
    shapes_list = []
    for object in single_data["target"]:
        bbox = list2xyxy([*object["pt1"], *object["pt2"]])
        #xywh_list = xyxy2wxwh(bbox)
        shapes_list.append({
            "label": object["category"],
            "points": [[bbox.x1, bbox.y1], [bbox.x2, bbox.y2]],
            "group_id": None,
            "description": "",
            "shape_type": "rectangle",
            "flags": {},
            "mask": None
        })
    return {
        "version": "5.5.0",
        "flags": {},
        "imagePath": img_fn,
        "shapes": shapes_list,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w
    }


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

    # basic option
    target_name, json_str, mode, default_suffix, resize, save_path, debug_mode  = \
        args.target, args.json, args.mode, args.suffix, args.resize, args.save_path, args.debug

    # image option
    jpg_quality, png_compress = args.jpg_quality, args.png_compressing

    # src type
    src_type = "video"

    # save_path valid check
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # 获取Logger
    logger = get_default_logger()
    set_default_logger(debug_mode, work_mode="frontend")

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
        # 图像模式下，data需要进行一定预处理
        # 将 num_frame 转换为实际起止时间戳，并整理标注。
        if raw_data["type"] in ("image-prediction","timelapse-prediction"):
            for i in range(len(data)):
                raw_anno = data[i]
                start_time, end_time = None, None
                if raw_data["type"] == "timelapse-prediction":
                    start_time = frame2ts(raw_anno["num_frame"],
                                        raw_data["basic_info"]["fps"])
                    end_time = frame2ts(raw_anno["num_frame"] + 1,
                                        raw_data["basic_info"]["fps"])
                target = []
                for (box, pred) in zip(raw_anno["boxes"], raw_anno["preds"]):
                    target.append(dict(pt1=box[:2], pt2=box[2:], category=pred))
                data[i].update(video_size=raw_data["anno_size"],
                            start_time=start_time,
                            end_time=end_time,
                            target=target)
            if raw_data["type"] == "image-prediction":
                src_type = "image"
    else:
        # target被作为视频解析。从参数构造单个使用的data。
        video_name = target_name
        data = [dict(start_time=args.start_time, end_time=args.end_time)]

    if src_type == "image":
        # TODO: 目前通过early-return分流，未来重构此部分结构。
        try:
            logger.start()
            for image in data:
                _, fname = os.path.split(image["img_filename"])
                full_path = os.path.join(save_path, fname)
                image_data = load_8bit_image(image["img_filename"])
                # 转换image标注为对应格式
                anno_dict = dict(
                    video_size=image_data.shape[:2][1::-1],
                    target=[
                        dict(pt1=b[:2], pt2=b[2:], category=c)
                        for (b, c) in zip(image["boxes"], image["preds"])
                    ])
                if args.with_bbox:
                    for target in anno_dict.get("target", []):
                        if not ("pt1" in target and "pt2" in target):
                            logger.warning(
                                f"lack pt1 or pt2 in dataline: {target}.")
                        pt1 = list(map(int, target["pt1"]))
                        pt2 = list(map(int, target["pt2"]))
                        image_data = cv2.rectangle(image_data,
                                                   pt1,
                                                   pt2,
                                                   color=[0, 0, 255],
                                                   thickness=2)
                # 保存图像到目标路径下
                save_img(image_data, full_path, args.jpg_quality,
                         args.png_compressing)
                logger.info(f"Saved: {full_path}")
                # 在有target的情况下，同时生成labelme风格的标注
                if args.with_annotation:
                    res_dict = generate_labelme(anno_dict, img_fn=full_path)
                    if res_dict:
                        anno_path = os.path.join(
                            save_path,
                            ".".join(full_path.split(".")[:-1]) + ".json")
                        with open(anno_path, mode="w", encoding="utf-8") as f:
                            json.dump(res_dict,
                                      f,
                                      ensure_ascii=False,
                                      indent=4)
                        logger.info(f"Saved: {anno_path}")
        finally:
            logger.stop()
        return
    video_loader = ThreadVideoLoader(OpenCVVideoWrapper,
                                     video_name,
                                     resize_option=resize,
                                     exp_option="real-time",
                                     resize_interpolation=cv2.INTER_LANCZOS4,
                                     debayer=args.debayer,
                                     debayer_pattern=args.debayer_pattern)

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
                if results is None:
                    logger.error(f"Fail to generate image for data: {target}.")
                    continue
                if args.with_bbox:
                    for target in single_data.get("target", []):
                        if not ("pt1" in target and "pt2" in target):
                            logger.warning(
                                f"lack pt1 or pt2 in dataline: {target}.")
                        pt1 = list(map(int, target["pt1"]))
                        pt2 = list(map(int, target["pt2"]))
                        results = cv2.rectangle(results,
                                                pt1,
                                                pt2,
                                                color=[0, 0, 255],
                                                thickness=2)
                if results is not None:
                    save_img(results, full_path, jpg_quality, png_compress)
                    logger.info(f"Saved: {full_path}")
                else:
                    logger.error("Error occured, got empty image.")
                # 在有target的情况下，同时生成labelme风格的标注
                if args.with_annotation:
                    res_dict = generate_labelme(single_data, img_fn=tgt_name)
                    if res_dict:
                        anno_path = os.path.join(
                            save_path,
                            ".".join(tgt_name.split(".")[:-1]) + ".json")
                        with open(anno_path, mode="w", encoding="utf-8") as f:
                            json.dump(res_dict,
                                      f,
                                      ensure_ascii=False,
                                      indent=4)
                        logger.info(f"Saved: {anno_path}")
            else:
                status_code = save_video_by_stream(video_loader,
                                                   video_loader.fps, full_path)
                if status_code == 0:
                    logger.info(f"Saved: {full_path}")
                else:
                    logger.error(
                        f"Error occured when writing the video to {full_path}."
                    )
    finally:
        logger.stop()


if __name__ == "__main__":
    main()
