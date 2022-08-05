from functools import partial

import numpy as np

from .utils import m3func

available_func = dict(max=partial(np.max, axis=0), m3func=m3func)


def init_stacker(name, cfg, exp_frame):
    if name == "SimpleStacker":
        return SimpleStacker()
    elif name == "MergeStacker":
        pfunc = cfg["pfunc"]
        if not pfunc in available_func:
            raise NameError(
                "Unsupported preprocessing function name: %s; Only %s are supported now."
                % (pfunc, available_func))
        func = available_func[pfunc]

        return MergeStacker(func, window_size=exp_frame)
    else:
        raise NameError("Undefined stacker name: %s" % (name))


class BaseStacker(object):
    def __init__(self, window_size=1):
        self.window_size = window_size

    def update(self):
        raise NotImplementedError(
            "This Method Is Not Implemented In This Base Class.")


class SimpleStacker(BaseStacker):
    def __init__(self, *args, **kwargs):
        # force window_size=1
        super().__init__(window_size=1)

    def update(self, video_reader, detector):
        """原始的栈更新方法: 从video_stack直接加载帧放入检测窗口内。 

        Args:
            video_stack (_type_): _description_
            detector (_type_): _description_
        """
        self.cur_frame = video_reader.pop(self.window_size)[0]
        detector.update(self.cur_frame)


class MergeStacker(BaseStacker):
    def __init__(self, func, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = func

    def update(self, video_reader, detector):
        """实验性的栈更新方法: 从video_stack加载若干帧，通过合并算法计算为一帧后放入检测窗口内

        Args:
            video_stack (_type_): _description_
            detector (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.cur_frame = self.func(video_reader.pop(self.window_size))
        detector.update(self.cur_frame)
