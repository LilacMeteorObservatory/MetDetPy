from functools import partial

import cv2
import numpy as np

from .VideoLoader import ThreadVideoReader

identity = lambda x: x

def generate_resize_func(resize):
    if resize:
        return partial(cv2.resize,
                       dsize=resize,
                       interpolation=cv2.INTER_LANCZOS4)
    return identity


def all_stacker(video, start_frame, end_frame, resize):
    """Load all frames to a mat(list, actually).

    Args:
        video (_type_): _description_
        start_frame (_type_): _description_
        end_frame (_type_): _description_

    Returns:
        _type_: _description_
    """
    pre_func = generate_resize_func(resize)
    iterations = end_frame - start_frame + 1
    mat = []
    video_reader = ThreadVideoReader(video,
                                     start_frame,
                                     iterations,
                                     pre_func,
                                     exp_frame=1,
                                     merge_func="max")
    try:
        video_reader.start()
        for i in range(iterations):
            mat.append(video_reader.pop())
    finally:
        video_reader.stop()

    return mat


def max_stacker(video, start_frame, end_frame, resize):

    pre_func = generate_resize_func(resize)
    iterations = end_frame - start_frame + 1
    video_reader = ThreadVideoReader(video,
                                     start_frame,
                                     iterations,
                                     pre_func,
                                     exp_frame=1,
                                     merge_func="max")
    try:
        video_reader.start()
        # Load first frame as the base frame.
        base_frame = video_reader.pop()
        for i in range(iterations - 1):
            new_frame = video_reader.pop()
            # stack: create
            base_frame = np.max([base_frame, new_frame], axis=0)
        return base_frame
    finally:
        video_reader.stop()
