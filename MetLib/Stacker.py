class BaseStacker(object):
    def __init__(self, frames=1):
        self.frames = frames

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
        detector.update(video_reader.pop(self.frames))
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
        detector.update([self.func(video_reader.pop(self.frames))])
        return detector