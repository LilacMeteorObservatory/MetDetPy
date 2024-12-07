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
import os
import tqdm
import cv2
import numpy as np

from MetLib.MetLog import get_default_logger
from MetLib.Model import YOLOModel
from MetLib.utils import ID2NAME, load_8bit_image, load_mask, pt_offset
from MetLib.VideoLoader import ThreadVideoLoader
from MetLib.VideoWrapper import OpenCVVideoWrapper
from MetLib.MetVisu import OpenCVMetVisu

SUPPORT_IMG_FORMAT = ["jpg", "png", "jpeg", "tiff", "tif", "bmp"]
SUPPORT_VIDEO_FORMAT = ["avi", "mp4", "mkv", "mpeg"]
CATE2COLOR_MAPPING = {
    "METEOR": [0, 255, 0],
    "PLANE/SATELLITE": [64, 64, 64],
    "RED_SPRITE": [0, 0, 255],
    "LIGHTNING": [128, 128, 128],
    "JET": [0, 0, 255],
    "RARE_SPRITE": [0, 0, 255],
    "SPACECRAFT": [255, 0, 255]
}

parser = argparse.ArgumentParser()
parser.add_argument("target", help="path to the img or video.")
parser.add_argument("--mask", help="path to the mask file.")
parser.add_argument("--model-path",
                    help="/path/to/the/model",
                    default="./weights/yolov5s.onnx")
parser.add_argument("--model-type",
                    help="type of the model. Support YOLO.",
                    default="YOLOModel")
parser.add_argument("--debayer",
                    help="apply debayer to the given image/video.",
                    action="store_true")
parser.add_argument("--visu",
                    "-V",
                    action="store_true",
                    help="show detect results.")
#parser.add_argument("--output",
#                    "-V",
#                    action="store_true",
#                    help="show detect results.")

args = parser.parse_args()

input_path = args.target
model_path = args.model_path
logger = get_default_logger()
model = YOLOModel(model_path,
                  dtype="float32",
                  nms=True,
                  warmup=True,
                  logger=logger)
suffix = input_path.split(".")[-1].lower()

if suffix in SUPPORT_IMG_FORMAT:
    # img mode
    img = load_8bit_image(input_path)
    if img is None:
        raise ValueError(f"Failed to load image file from {input_path}.")
    mask = load_mask(args.mask, list(img.shape[1::-1]))
    img = img * mask
    boxes, preds = model.forward(img)
    print(list(zip(boxes, [ID2NAME[np.argmax(pred)] for pred in preds])))
    if args.visu:
        for b, p in zip(boxes, preds):
            x1, y1, x2, y2 = b
            img = cv2.rectangle(img, (x1, y1), (x2, y2), [255, 0, 0], 2)
        h, w, c = img.shape
        w = w / h * 720
        img = cv2.resize(img, (int(w), 720))
        cv2.imshow("show", img)
        cv2.waitKey(0)
elif suffix in SUPPORT_VIDEO_FORMAT:
    # video mode
    video = ThreadVideoLoader(OpenCVVideoWrapper,
                              input_path,
                              mask_name=args.mask,
                              exp_option="real-time",
                              resize_option=model.w)
    # 可视化参数组
    visu_param = dict(
        active_meteors=["draw", {
            "type": "rectangle",
            "color": "as-input"
        }],
        score_bg=[
            "draw", {
                "type": "rectangle",
                "position": "as-input",
                "color": "as-input",
                "thickness": -1
            }
        ],
        score_text=["text", {
            "position": "as-input",
            "color": "white"
        }])
    tot_frames = video.iterations
    video.start()
    visual_manager = OpenCVMetVisu(exp_time=1,
                                   resolution=video.runtime_size,
                                   flag=args.visu,
                                   visu_param_list=[visu_param])
    results = []
    for i in tqdm.tqdm(range(tot_frames)):
        img = video.pop()
        if img is None: continue
        # TODO: 整合debayer到预处理步骤
        if args.debayer:
            img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                               cv2.COLOR_BAYER_BGGR2BGR,
                               dstCn=3)
        boxes, preds = model.forward(img)
        results.append({
            "num_frame":
            i,
            "boxes":
            boxes,
            "preds": [ID2NAME[int(np.argmax(pred))] for pred in preds]
        })
        if args.visu:
            visu_info = dict(main_bg=img,
                             timestamp=[{
                                 "text": f"{i}/{tot_frames} imgs"
                             }],
                             active_meteors=[],
                             score_bg=[],
                             score_text=[])
            for b, p in zip(boxes, preds):
                cate_id = int(np.argmax(p))
                color = CATE2COLOR_MAPPING.get(ID2NAME[cate_id], [64, 64, 64])
                x1, y1, x2, y2 = b
                text = f"{ID2NAME[cate_id]}:{np.max(p):2f}"
                visu_info["active_meteors"].append({
                    "position": ((x1, y1), (x2, y2)),
                    "color":
                    color
                })
                visu_info["score_bg"].append({
                    "position":
                    ((x1, y1), pt_offset((x1, y1), (10 * len(text), -15))),
                    "color":
                    color
                })
                visu_info["score_text"].append({
                    "position":
                    pt_offset((x1, y1), (0, -2)),
                    "text":
                    text
                })
            visual_manager.display_a_frame(visu_info)
            if visual_manager.manual_stop:
                logger.info('Manual interrupt signal detected.')
                break
else:
    raise NotImplementedError(
        f"Unsupport file suffix \"{suffix}\". For now this only support {SUPPORT_VIDEO_FORMAT} and {SUPPORT_IMG_FORMAT}."
    )
