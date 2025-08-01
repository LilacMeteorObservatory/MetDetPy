from typing import Callable

from .Detector import (BaseDetector, ClassicDetector, DiffAreaGuidingDetecor,
                       M3Detector, MLDetector)
from .VideoLoader import (ProcessVideoLoader, ThreadVideoLoader,
                          VanillaVideoLoader)
from .VideoWrapper import BaseVideoWrapper, OpenCVVideoWrapper, PyAVVideoWrapper


def get_xxx(name: str, all: list[type]) -> Callable[[str], type]:
    name2class = {cls.__name__: cls for cls in all}

    def core(class_name: str) -> type:
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

get_loader = get_xxx("loader", available_loaders)
get_wrapper = get_xxx("wrapper", available_wrappers)
get_detector = get_xxx("detector", all=available_detectors)
