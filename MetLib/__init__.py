from typing import Callable, Any, Type

from .VideoLoader import ThreadVideoLoader, VanillaVideoLoader, ProcessVideoLoader
from .VideoWarpper import OpenCVVideoWarpper
from .Detector import M3Detector, ClassicDetector


def get_xxx(name, all) -> Callable[[str],Type]:
    name2class = {cls.__name__: cls for cls in all}

    def core(class_name) -> Type:
        if not class_name in name2class:
            raise Exception(f"No class named {class_name} for {name}.")
        return name2class[class_name]

    return core


available_loaders = [VanillaVideoLoader, ThreadVideoLoader,ProcessVideoLoader]
available_warppers = [OpenCVVideoWarpper]
available_detectors = [M3Detector, ClassicDetector]

get_loader = get_xxx("loader", available_loaders)
get_warpper = get_xxx("warpper", available_warppers)
get_detector= get_xxx("detector",all=available_detectors)