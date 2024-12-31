"""
VideoWrapper wraps video-related API, so that VideoLoader can use unified 
API to obtain metadata and frame data.

VideoWrapper对读取视频的API进行初步包装, 使VideoLoader能够使用统一的接口获取元数据及帧数据。
"""

from abc import ABCMeta, abstractmethod
from typing import Union

import cv2
import numpy as np


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
    def fps(self) -> float:
        pass

    @property
    @abstractmethod
    def num_frames(self) -> int:
        pass

    @property
    @abstractmethod
    def size(self) -> list:
        pass

    @property
    @abstractmethod
    def backend_name(self) -> str:
        pass

    @abstractmethod
    def set_to(self, frame):
        pass

    def release(self):
        pass

    @abstractmethod
    def read(self) -> Union[tuple, list]:
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

    @property
    def backend_name(self):
        return self.video.getBackendName()

    def read(self):
        return self.video.read()

    def release(self):
        self.video.release()

    def set_to(self, frame: int):
        """设置当前指针位置。
        """
        # TODO: 对于部分编码损坏的视频，set_to会耗时很长，并且后续会读取失败。应当做对应处置。
        # TODO 2: 对于不能set_to的，能否向后继续跳转？
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame)


if False:
    # the following code is not ready for use.
    # it is an experimental usage of pyav reading video.
    import av
    from .utils import frame2time

    class PyAVVideoWrapper(BaseVideoWrapper):
        """VideoWrapper for pyav-based video loader.

        Args:
            video_name (str): The video filename.

        Raises:
            FileNotFoundError: triggered when the video file can not be opened. 
        """

        def __init__(self, video_name: str) -> None:
            self.container = av.open(video_name)
            self.video = self.container.streams.video[0]
            # TODO: 合法性检查?

        @property
        def fps(self):
            return float(self.video.base_rate)

        @property
        def backend_name(self):
            return "FFmpeg"

        @property
        def num_frames(self):
            return (self.video.duration * self.video.time_base * self.fps) + 1

        @property
        def size(self):
            return [int(self.video.width), int(self.video.height)]

        def read(self):
            try:
                while True:
                    frame = self.container.demux(video=0).__next__().decode()
                    if len(frame) == 0:
                        continue
                    if len(frame) == 1:
                        image = cv2.cvtColor(frame[0].to_ndarray(),
                                             cv2.COLOR_RGB2BGR)
                    else:
                        image = cv2.cvtColor(frame[0].to_ndarray(),
                                             cv2.COLOR_YUV2BGR_I420)
                    return True, image
            except av.error.EOFError:
                return False, None

        def release(self):
            self.container.close()

        def set_to(self, frame: int):
            """设置当前指针位置。
            """
            self.container.seek(int(
                frame2time(frame, self.fps) / self.video.time_base),
                                any_frame=True,
                                stream=self.video)
