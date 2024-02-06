"""用于单次或批量运行模型的脚本。
可用于流星雨延时检测。
"""
import argparse
import numpy as np
import cv2
from MetLib.Model import YOLOModel
from MetLib.utils import ID2NAME, load_8bit_image
from MetLib.MetLog import get_default_logger

parser = argparse.ArgumentParser()
parser.add_argument("image", help="/path/to/the/img")
parser.add_argument("--model-path",
                    help="/path/to/the/model",
                    default="./weights/yolov5s.onnx")
parser.add_argument("--model-type",
                    help="type of the model. Support YOLO.",
                    default="YOLOModel")
parser.add_argument("--visu",
                    "-V",
                    action="store_true",
                    help="show detect results.")

args = parser.parse_args()

input_path = args.image
model_path = args.model_path

logger = get_default_logger()

model = YOLOModel(model_path,
                  dtype="float32",
                  nms=True,
                  warmup=True,
                  logger=logger)

img = load_8bit_image(input_path)

boxes, preds = model.forward(img)
print(boxes, preds)
if args.visu:
    for b, p in zip(boxes, preds):
        x1, y1, x2, y2 = b
        img = cv2.rectangle(img, (x1, y1), (x2, y2), [255, 0, 0], 2)
        print(ID2NAME[np.argmax(preds)])
    cv2.imshow("show", img)
    cv2.waitKey(0)