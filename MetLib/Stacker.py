from datetime import datetime

import numpy as np
import tqdm

from .VideoLoader import ThreadVideoReader

identity = lambda x: x

def dt2ts(dt: datetime) -> float:
    """
    Transfer a datetime.datetime object to float.
    
    I implement this only because datetime.timestamp() 
    seems does not support my data.
    
    (Maybe because it is default to be started from 1990-01-01)
    (Also, this function does not support time that longer than 24h.)

    Args:
        dt (datetime.datetime): the time object.

    Returns:
        float: time (in second).
    """
    return dt.hour * 60**2 + dt.minute * 60**1 + dt.second + dt.microsecond / 1e6

def max_stacker(video, start_frame, end_frame, pre_func=identity):
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

def time2frame(time: str, fps: float) -> int:
    """Transfer a utc time string into the frame num.

    Args:
        time (str): UTC time string.
        fps (float): frame per second of the video.

    Returns:
        int: the corresponding frame num of the input time.
        
    Example:
        time2frame("00:00:02.56",25) -> 64(=(2+(56/100))*25))
    """
    dt_time = datetime.strptime(time, "%H:%M:%S.%f")
    return int(dt2ts(dt_time) * fps)
