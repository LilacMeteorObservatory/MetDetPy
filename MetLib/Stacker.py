import numpy as np
from typing import Optional


def all_stacker(video_loader,
                start_frame=None,
                end_frame=None,
                logger=None) -> list:
    """Load all frames to a mat(list, actually).

    Args:
        video (_type_): _description_
        start_frame (_type_): _description_
        end_frame (_type_): _description_

    Returns:
        _type_: _description_
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


def max_stacker(video_loader,
                start_frame=None,
                end_frame=None,
                logger=None) -> Optional[np.ndarray]:
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