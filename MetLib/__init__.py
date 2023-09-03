from typing import Callable

from .VideoLoader import ThreadVideoLoader, VanillaVideoLoader
from .VideoWarpper import OpenCVVideoWarpper


def get_xxx(name, all) -> Callable:
    name2class = {cls.__name__: cls for cls in all}

    def core(class_name) -> Callable:
        if not class_name in name2class:
            raise Exception(f"No class named {class_name} for {name}.")
        return name2class[class_name]

    return core


available_loaders = [VanillaVideoLoader, ThreadVideoLoader]
available_warppers = [OpenCVVideoWarpper]

get_loader = get_xxx("loader", available_loaders)
get_warpper = get_xxx("warpper", available_warppers)