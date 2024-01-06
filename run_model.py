"""用于单次或批量运行模型的脚本。
可用于流星雨延时检测。
"""
import argparse

from MetLib.Model import YOLOModel
from MetLib.utils import ID2NAME,load_8bit_image
from MetLib.MetLog import get_default_logger

model_path = "./weights/yolov5s.onnx"
input_path = "path/to/my/img"

logger = get_default_logger()

model = YOLOModel(model_path,dtype="float32",nms=True,warmup=True,logger=logger)

img = load_8bit_image(input_path)

results = model.forward(img)

print(results)