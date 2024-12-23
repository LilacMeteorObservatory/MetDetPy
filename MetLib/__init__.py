from typing import Any, Callable, Type

from .Detector import (ClassicDetector, DiffAreaGuidingDetecor, M3Detector,
                       MLDetector)
from .VideoLoader import (ProcessVideoLoader, ThreadVideoLoader,
                          VanillaVideoLoader)
from .VideoWrapper import OpenCVVideoWrapper


def get_xxx(name, all) -> Callable[[str], Type]:
    name2class = {cls.__name__: cls for cls in all}

    def core(class_name) -> Type:
        if not class_name in name2class:
            raise Exception(f"No class named {class_name} for {name}.")
        return name2class[class_name]

    return core


available_loaders = [VanillaVideoLoader, ThreadVideoLoader, ProcessVideoLoader]
available_wrappers = [OpenCVVideoWrapper]
available_detectors = [
    M3Detector, ClassicDetector, MLDetector, DiffAreaGuidingDetecor
]

get_loader = get_xxx("loader", available_loaders)
get_wrapper = get_xxx("wrapper", available_wrappers)
get_detector = get_xxx("detector", all=available_detectors)
