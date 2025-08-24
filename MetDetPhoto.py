"""
利用目标检测模型，从单张图像或图像序列批量检测流星的工具。

适用数据：
* 单张图像
* TODO: 批量图像（文件夹）
* 延时视频（照片构成的序列）

## 支持的保存格式
1. 带有标注框的图像
2. 图像和标注文件
3. MDRF形式的摘要
"""
import argparse
import json
import os
from typing import cast

import cv2
import numpy as np
import tqdm
from numpy.typing import NDArray

from MetLib.fileio import load_8bit_image, load_mask, save_path_handler
from MetLib.metlog import get_default_logger, set_default_logger
from MetLib.metstruct import MDRF, MockVideoObject, SingleImgRecord
from MetLib.metvisu import (BaseVisuAttrs, ColorTuple, DrawRectVisu,
                            OpenCVMetVisu, SquareColorPair, TextColorPair,
                            TextVisu)
from MetLib.model import YOLOModel
from MetLib.utils import ID2NAME, VERSION, parse_resize_param, pt_offset
from MetLib.videoloader import ThreadVideoLoader
from MetLib.videowrapper import OpenCVVideoWrapper

SUPPORT_IMG_FORMAT = ["jpg", "png", "jpeg", "tiff", "tif", "bmp"]
SUPPORT_VIDEO_FORMAT = ["avi", "mp4", "mkv", "mpeg"]
EXCLUDE_LIST = ["PLANE/SATELLITE", "BUGS"]
DEFAULT_COLOR = (64, 64, 64)
DEFAULT_VISUAL_WINDOW_SIZE = [960, 540]
CATE2COLOR_MAPPING: dict[str, ColorTuple] = {
    "METEOR": (0, 255, 0),
    "PLANE/SATELLITE": DEFAULT_COLOR,
    "RED_SPRITE": (0, 0, 255),
    "LIGHTNING": (128, 128, 128),
    "JET": (0, 0, 255),
    "RARE_SPRITE": (0, 0, 255),
    "SPACECRAFT": (255, 0, 255)
}

# 可视化参数组
visu_param: list[BaseVisuAttrs] = [
    TextVisu("timestamp", position="left-bottom", color="white"),
    DrawRectVisu(
        name="activate_meteors",
        color="as-input",
    ),
    DrawRectVisu(name="score_bg", thickness=-1),
    TextVisu(name="score_text", color="white")
]


def construct_visu_info(boxes: NDArray[np.int_],
                        preds: NDArray[np.float64],
                        watermark_text: str = ""):
    """构建可视化信息返回串。

    Args:
        img (np.ndarray): background image
        boxes (list[np.ndarray]): boxes
        preds (list[np.ndarray]): pred
        watermark_text (str, optional): watermark. Defaults to "".

    Returns:
        dict: visu_info that can be loaded by MetVisu directly.
    """
    active_meteors: list[SquareColorPair] = []
    score_bg: list[SquareColorPair] = []
    score_text: list[TextColorPair] = []
    for b, p in zip(boxes, preds):
        cate_id = int(np.argmax(p))
        color = CATE2COLOR_MAPPING.get(ID2NAME[cate_id], DEFAULT_COLOR)
        x1, y1, x2, y2 = b
        text = f"{ID2NAME[cate_id]}:{np.max(p):2f}"
        active_meteors.append(
            SquareColorPair(([x1, y1], [x2, y2]), color=color))
        score_bg.append(
            SquareColorPair(
                ([x1, y1], pt_offset((x1, y1), (10 * len(text), -15))),
                color=color))
        score_text.append(
            TextColorPair(text, position=pt_offset((x1, y1), (0, -2))))
    visu_info: list[BaseVisuAttrs] = [
        TextVisu("timestamp", text_list=[TextColorPair(watermark_text)]),
        DrawRectVisu("activate_meteors", pair_list=active_meteors),
        DrawRectVisu("score_bg", pair_list=score_bg),
        TextVisu("score_text", text_list=score_text)
    ]
    return visu_info


parser = argparse.ArgumentParser()
parser.add_argument("target", help="path to the img or video.")
parser.add_argument("--mask", help="path to the mask file.")
parser.add_argument("--model-path",
                    help="/path/to/the/model",
                    default="./weights/yolov5s_v2.onnx")
parser.add_argument("--exclude-noise", action="store_true")
parser.add_argument("--model-type",
                    help="type of the model. Support YOLO.",
                    default="YOLOModel")
parser.add_argument("--debayer",
                    help="apply debayer to the given image/video.",
                    action="store_true")
parser.add_argument("--debayer-pattern",
                    help="debayer pattern, like RGGB or BGGR.")
parser.add_argument("--scale",
                    "-M",
                    type=int,
                    default=2,
                    help="multiscale num.")
parser.add_argument("--partition",
                    "-P",
                    type=int,
                    default=2,
                    help="partition in pyramid.")
parser.add_argument("--visu",
                    "-V",
                    action="store_true",
                    help="show detect results.")
parser.add_argument("--visu-resolution",
                    "-R",
                    type=str,
                    help="detect results showing resolution.")
parser.add_argument("--save-path", "-S", type=str, help="save path for MDRF.")
parser.add_argument("--debug", "-D", action="store_true", help="debug mode.")

args = parser.parse_args()

input_path = args.target
model_path = args.model_path
visu_resolution = parse_resize_param(
    args.visu_resolution, DEFAULT_VISUAL_WINDOW_SIZE
) if args.visu_resolution else DEFAULT_VISUAL_WINDOW_SIZE

set_default_logger(debug_mode=args.debug, work_mode="frontend")
logger = get_default_logger()

model = YOLOModel(model_path,
                  dtype="float32",
                  nms=True,
                  warmup=True,
                  logger=logger,
                  multiscale_pred=args.scale,
                  multiscale_partition=args.partition)
logger.start()
valid_flag = False
results: list[SingleImgRecord] = []
video = None
try:
    if os.path.isdir(input_path):
        # img folder mode
        img_list = [
            os.path.join(input_path, x)
            for x in cast(list[str], os.listdir(input_path))
            if x.split(".")[-1].lower() in SUPPORT_IMG_FORMAT
        ]
        visual_manager = OpenCVMetVisu(exp_time=1,
                                       resolution=visu_resolution,
                                       flag=args.visu,
                                       visu_param_list=visu_param)

        # temp fix: mock video object
        video = MockVideoObject(image_folder=input_path)
        for img_path in tqdm.tqdm(img_list):
            img = load_8bit_image(img_path)
            if img is None:
                logger.error(f"Failed to load image file from {input_path}.")
                continue
            mask = load_mask(args.mask, list(img.shape[1::-1]))
            img = img * mask
            boxes, preds = model.forward(img)
            if args.visu:
                visu_info = construct_visu_info(boxes,
                                                preds,
                                                watermark_text=img_path)
                visual_manager.display_a_frame(img, visu_info)
                if visual_manager.manual_stop:
                    logger.info('Manual interrupt signal detected.')
                    break
            if len(boxes) > 0:
                results.append(
                    SingleImgRecord(boxes=[list(map(int, x)) for x in boxes],
                                    preds=[
                                        ID2NAME[int(np.argmax(pred))]
                                        for pred in preds
                                    ],
                                    prob=[
                                        f"{pred[int(np.argmax(pred))]:.2f}"
                                        for pred in preds
                                    ],
                                    img_filename=img_path))

    elif os.path.isfile(input_path):
        suffix = input_path.split(".")[-1].lower()

        if suffix in SUPPORT_IMG_FORMAT:
            # img mode
            # temp fix: mock video object
            video = MockVideoObject(image_folder=input_path)
            img = load_8bit_image(input_path)
            if img is None:
                raise ValueError(
                    f"Failed to load image file from {input_path}.")
            mask = load_mask(args.mask, list(img.shape[1::-1]))
            img = img * mask
            visual_manager = OpenCVMetVisu(exp_time=1,
                                           resolution=visu_resolution,
                                           flag=args.visu,
                                           visu_param_list=visu_param)
            boxes, preds = model.forward(img)

            results = [
                SingleImgRecord(
                    boxes=[list(map(int, x)) for x in boxes],
                    preds=[ID2NAME[int(np.argmax(pred))] for pred in preds],
                    prob=[
                        f"{pred[int(np.argmax(pred))]:.2f}" for pred in preds
                    ],
                    img_filename=input_path)
            ]
            print(boxes, preds)
            #preds = [ID2NAME[int(np.argmax(pred))] for pred in preds]
            if args.visu:
                visu_info = construct_visu_info(boxes,
                                                preds,
                                                watermark_text=input_path)
                visual_manager.display_a_frame(img, visu_info)
                cv2.waitKey(0)
        elif suffix in SUPPORT_VIDEO_FORMAT:
            # video mode
            video = ThreadVideoLoader(OpenCVVideoWrapper,
                                      input_path,
                                      mask_name=args.mask,
                                      exp_option="real-time",
                                      debayer=args.debayer,
                                      debayer_pattern=args.debayer_pattern)
            tot_frames = video.iterations
            video.start()
            visual_manager = OpenCVMetVisu(exp_time=1,
                                           resolution=visu_resolution,
                                           flag=args.visu,
                                           visu_param_list=visu_param)
            results = []
            for i in tqdm.tqdm(range(tot_frames)):
                img = video.pop()
                if img is None: continue
                boxes, probs = model.forward(img)
                if args.visu:
                    visu_info = construct_visu_info(
                        boxes, probs, watermark_text=f"{i}/{tot_frames} imgs")
                    visual_manager.display_a_frame(img, visu_info)
                    if visual_manager.manual_stop:
                        logger.info('Manual interrupt signal detected.')
                        break
                # TODO: fix this in the future.
                preds = [ID2NAME[int(np.argmax(pred))] for pred in probs]
                if args.exclude_noise:
                    selected_id = [
                        i for i, pred in enumerate(preds)
                        if pred not in EXCLUDE_LIST
                    ]
                    boxes = [boxes[i] for i in selected_id]
                    preds = [preds[i] for i in selected_id]
                if len(boxes) > 0:
                    results.append(
                        SingleImgRecord(
                            boxes=[list(map(int, x)) for x in boxes],
                            preds=preds,
                            prob=[
                                f"{pred[int(np.argmax(pred))]:.2f}"
                                for pred in probs
                            ],
                            num_frame=i))
        else:
            raise NotImplementedError(
                f"Unsupport file suffix \"{suffix}\". For now this only support {SUPPORT_VIDEO_FORMAT} and {SUPPORT_IMG_FORMAT}."
            )
    else:
        raise FileNotFoundError(f"File {input_path} does not exist!")
    valid_flag = True
except Exception as e:
    logger.error(e.__repr__())
finally:
    logger.stop()

# 保存结果
if valid_flag and args.save_path and video is not None:
    fin_result = MDRF(version=VERSION,
                      basic_info=video.summary(),
                      config=None,
                      type="image-prediction" if isinstance(
                          video, MockVideoObject) else "timelapse-prediction",
                      anno_size=video.summary().resolution,
                      results=results)
    with open(save_path_handler(args.save_path, input_path, ext="json"),
              mode="w",
              encoding="utf-8") as f:
        json.dump(fin_result.to_dict(), f, ensure_ascii=False, indent=4)
