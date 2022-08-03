from matplotlib.style import available
import numpy as np
from .utils import m3func, init_exp_time

available_func = dict(max=np.max, m3func=m3func)
auto_framer = dict(realtime=1, )


def init_stacker(name, cfg, video, mask, fps):
    if name == "SimpleStacker":
        return SimpleStacker(), 1 / fps
    elif name == "MergeStacker":
        pfunc, exp_time = cfg["pfunc"], cfg["exp_time"]
        if not pfunc in available_func:
            raise NameError(
                "Unsupported preprocessing function name: %s; Only %s are supported now."
                % (pfunc, available_func))
        func = available(pfunc)
        exp_time = init_exp_time(exp_time, video, mask)

        return MergeStacker(func, window_size=int(exp_time * fps)), exp_time
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
        super().__init__(*args, **kwargs)

    def update(self, video_reader, detector):
        """原始的栈更新方法: 从video_stack直接加载帧放入检测窗口内。 

        Args:
            video_stack (_type_): _description_
            detector (_type_): _description_
        """
        detector.update(video_reader.pop(self.window_size))
        return detector


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
        detector.update([self.func(video_reader.pop(self.window_size))])
        return detector