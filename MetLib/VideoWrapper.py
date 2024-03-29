"""
VideoWrapper wraps video-related API, so that VideoLoader can use unified 
API to obtain metadata and frame data.

VideoWrapper对读取视频的API进行初步包装, 使VideoLoader能够使用统一的接口获取元数据及帧数据。
"""

import cv2
from typing import Union
from abc import ABCMeta, abstractmethod

class BaseVideoWrapper(metaclass=ABCMeta):
    """
    ## BaseVideoWrapper
    Abstract Base Class of VideoWrapper. Inherit this to implement your Videowrapper.

    ### What your VideoWrapper should support:
    #### Property:
    fps -> Union[int, float] # frame per second

    num_frames -> int # total num of frames
    
    size -> Union[list, tuple] # [width, height] of the video
    
    #### Method:

    set_to(frame: int) # set current frame position(?)

    release() # release fp

    read()-> ret_code, frame # load a frame from Video

    """

    def __init__(self, video_name) -> None:
        pass

    @property
    @abstractmethod
    def fps(self)->float:
        pass

    @property
    @abstractmethod
    def num_frames(self)->int:
        pass

    @property
    @abstractmethod
    def size(self)->list:
        pass

    @abstractmethod
    def set_to(self, frame):
        pass

    def release(self):
        pass

    @abstractmethod
    def read(self)->Union[tuple, list]:
        pass


class OpenCVVideoWrapper(BaseVideoWrapper):
    """VideoWrapper for opencv-based video loader (cv2.VideoCapture)

    Args:
        video_name (str): The video filename.

    Raises:
        FileNotFoundError: triggered when the video file can not be opened. 
    """

    def __init__(self, video_name: str) -> None:
        self.video = cv2.VideoCapture(video_name)
        if (self.video is None) or (not self.video.isOpened()):
            raise FileNotFoundError(
                f"The video \"{video_name}\" cannot be opened as a supported video format."
            )

    @property
    def fps(self):
        return self.video.get(cv2.CAP_PROP_FPS)

    @property
    def num_frames(self):
        return int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def size(self):
        return [
            int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ]

    def read(self):
        return self.video.read()

    def release(self):
        self.video.release()

    def set_to(self, frame: int):
        """设置当前指针位置。
        """
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame)