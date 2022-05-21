class BaseStacker(object):
    def __init__(self, frames=1):
        self.frames = 1

    def update(self, video_stack, detect_stack):
        # 原始的栈更新方法
        # 从video_stack直接加载帧放入检测窗口内
        clipped_stack, video_stack = video_stack[:self.frames], video_stack[
            self.frames:]
        detect_stack.append(clipped_stack)
        return video_stack, detect_stack


class StackerMerger(BaseStacker):
    def __init__(self, func, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = func

    def update(self, video_stack, detect_stack):
        # 实验性的栈更新方法
        # 从video_stack加载若干帧，通过合并算法计算为一帧后放入检测窗口内
        detect_stack.pop(0)
        clipped_stack, video_stack = video_stack[:self.frames], video_stack[
            self.frames:]
        clipped_merged = self.func(clipped_stack)
        detect_stack.append(clipped_merged)
        return video_stack, detect_stack
