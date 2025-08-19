from typing import Callable, TypeVar

from .Detector import (BaseDetector, ClassicDetector, DiffAreaGuidingDetecor,
                       M3Detector, MLDetector)
from .videoloader import (ProcessVideoLoader, ThreadVideoLoader,
                          VanillaVideoLoader)
from .videowrapper import BaseVideoWrapper, OpenCVVideoWrapper, PyAVVideoWrapper

from .videowriter import BaseVideoWriter, OpenCVVideoWriter, PyAVVideoWriter

from .model import YOLOModel

T = TypeVar("T", type[VanillaVideoLoader], type[BaseVideoWrapper],
            type[BaseDetector], type[BaseVideoWriter], type[YOLOModel])


def get_xxx(name: str, all: list[T]) -> Callable[[str], T]:
    name2class = {cls.__name__: cls for cls in all}

    def core(class_name: str):
        if not class_name in name2class:
            raise Exception(f"No class named {class_name} for {name}.")
        return name2class[class_name]

    return core


available_loaders: list[type[VanillaVideoLoader]] = [
    VanillaVideoLoader, ThreadVideoLoader, ProcessVideoLoader
]
available_wrappers: list[type[BaseVideoWrapper]] = [
    OpenCVVideoWrapper, PyAVVideoWrapper
]
available_detectors: list[type[BaseDetector]] = [
    M3Detector, ClassicDetector, MLDetector, DiffAreaGuidingDetecor
]

available_writers: list[type[BaseVideoWriter]] = [
    BaseVideoWriter, OpenCVVideoWriter, PyAVVideoWriter
]

available_models_list = [YOLOModel]

get_loader = get_xxx("loader", available_loaders)
get_wrapper = get_xxx("wrapper", available_wrappers)
get_detector = get_xxx("detector", available_detectors)
get_writer = get_xxx("writer", available_writers)
get_model = get_xxx("model", available_models_list)
