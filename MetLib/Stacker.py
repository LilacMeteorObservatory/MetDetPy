class BaseStacker(object):
    def __init__(self, frames=1):
        self.frames = frames

    def update(self, video_stack, detect_stack, force=False):
        flag = True
        if (len(video_stack) < self.frames) and (not force):
            return False, video_stack, detect_stack
        video_stack, detect_stack = self._update(video_stack, detect_stack)
        return flag, video_stack, detect_stack

    def _update(self, video_stack, detect_stack):
        raise NotImplementedError(
            "This Method Is Not Implemented In This Base Class.")


class SimpleStacker(BaseStacker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _update(self, video_stack, detect_stack):
        """原始的栈更新方法: 从video_stack直接加载帧放入检测窗口内。 

        Args:
            video_stack (_type_): _description_
            detect_stack (_type_): _description_
        """
        clipped_stack, video_stack = video_stack[:self.frames], video_stack[
            self.frames:]
        detect_stack.update(clipped_stack)
        return video_stack, detect_stack


class MergeStacker(BaseStacker):
    def __init__(self, func, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = func

    def _update(self, video_stack, detect_stack):
        """实验性的栈更新方法: 从video_stack加载若干帧，通过合并算法计算为一帧后放入检测窗口内

        Args:
            video_stack (_type_): _description_
            detect_stack (_type_): _description_

        Returns:
            _type_: _description_
        """
        clipped_stack, video_stack = video_stack[:self.frames], video_stack[
            self.frames:]
        clipped_merged = self.func(clipped_stack)
        detect_stack.update([clipped_merged])
        return video_stack, detect_stack