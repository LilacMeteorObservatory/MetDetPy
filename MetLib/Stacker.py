from logging import Logger
from typing import Optional

import numpy as np

from .VideoLoader import BaseVideoLoader, VanillaVideoLoader


def all_stacker(video_loader: BaseVideoLoader,
                start_frame: Optional[int]=None,
                end_frame: Optional[int]=None,
                logger: Optional[Logger]=None) -> list:
    """ Load all frames to a matrix(list, actually).

    Args:
        video_loader (BaseVideoLoader): initialized video loader.
        start_frame (Optional[int], optional): the start frame of the stacker. Defaults to None.
        end_frame (Optional[int], optional): the end frame of the stacker.. Defaults to None.
        logger (Optional[Logger], optional): a logging.Logger object for logging use. Defaults to None.

    Returns:
        list: a list containing all frames.
    """
    if start_frame != None or end_frame != None:
        video_loader.reset(start_frame=start_frame, end_frame=end_frame)
    mat = []
    try:
        video_loader.start()
        for i in range(video_loader.iterations):
            mat.append(video_loader.pop())
    except Exception as e:
        if logger:
            logger.error(e.__repr__())
        return mat
    finally:
        video_loader.stop()

    return mat

def max_stacker(video_loader: VanillaVideoLoader,
                start_frame: Optional[int]=None,
                end_frame: Optional[int]=None,
                logger: Optional[Logger]=None) -> Optional[np.ndarray]:
    """Stack frames within range and return a stacked image.

    Args:
        video_loader (BaseVideoLoader): initialized video loader.
        start_frame (Optional[int], optional): the start frame of the stacker. Defaults to None.
        end_frame (Optional[int], optional): the end frame of the stacker.. Defaults to None.
        logger (Optional[Logger], optional): a logging.Logger object for logging use. Defaults to None.

    Returns:
        Optional[np.ndarray]: the stacked image. If there is no frames to stack, return None.
    """
    if start_frame != None or end_frame != None:
        video_loader.reset(start_frame=start_frame, end_frame=end_frame)
    base_frame = None
    try:
        video_loader.start()
        # Load first frame as the base frame.
        base_frame = video_loader.pop()
        for i in range(video_loader.iterations - 1):
            if video_loader.stopped:
                break
            new_frame = video_loader.pop()
            assert base_frame.shape == new_frame.shape, "Expect new " + \
                    f"frame has the same shape as the base frame ({base_frame.shape}), " + \
                    f"but {new_frame.shape} got."
            base_frame = np.max([base_frame, new_frame], axis=0)
        return base_frame
    except Exception as e:
        if logger:
            logger.error(e.__repr__())
        return base_frame
    finally:
        video_loader.stop()