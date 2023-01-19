from functools import partial

import numpy as np
from .VideoLoader import ThreadVideoReader 
from .utils import m3func, mix_max_median_stacker

identity = lambda x:x

class ImgStacker(object):
    def __init__(self) -> None:
        pass

# Q2：有关pop的代码能否优化（注意关注stacker等涉及到与VideoLoader交互的模块）


def max_img_stacker(video, start_frame, end_frame, pre_func=identity):
    iterations = end_frame - start_frame + 1
    video_reader = ThreadVideoReader(video, iterations, pre_func)
    video_reader.start()
    # Load first frame as the base frame.
    base_frame = video_reader.pop()
    for i in range(iterations - 1):
        new_frame = video_reader.pop()
        # stack: create
        base_frame = np.max([base_frame,new_frame], axis=0)
    return base_frame