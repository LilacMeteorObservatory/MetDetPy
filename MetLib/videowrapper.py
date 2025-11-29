"""
VideoWrapper wraps video-related API, so that VideoLoader can use unified 
API to obtain metadata and frame data.

VideoWrapper对读取视频的API进行初步包装, 使VideoLoader能够使用统一的接口获取元数据及帧数据。
"""

import os
from abc import ABCMeta, abstractmethod
from typing import Optional

import av
import av.error
import cv2
from cv2.typing import MatLike

from .utils import frame2time, time2frame


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

    def __init__(self, video_name: str) -> None:
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
    def size(self) -> list[int]:
        pass

    @property
    @abstractmethod
    def backend_name(self) -> str:
        pass

    @abstractmethod
    def set_to(self, frame_num: int) -> bool:
        pass

    def force_set_to(self, frame_num: int) -> bool:
        """逐帧索引，直接跳转的降级方案。"""
        return self.set_to(frame_num)

    @abstractmethod
    def get_video_pos(self) -> int:
        pass

    def release(self):
        pass

    @abstractmethod
    def read(self) -> tuple[bool, Optional[MatLike]]:
        pass


class OpenCVVideoWrapper(BaseVideoWrapper):
    """VideoWrapper for opencv-based video loader (cv2.VideoCapture)

    Args:
        video_name (str): The video filename.

    Raises:
        FileNotFoundError: triggered when the video file can not be opened. 
    """

    def __init__(self, video_name: str) -> None:
        self.video = cv2.VideoCapture(video_name, cv2.CAP_FFMPEG)
        if not self.video.isOpened():
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
        return f"{self.__class__.__name__}({self.video.getBackendName()})"

    def read(self):
        return self.video.read()

    def release(self):
        self.video.release()

    def set_to(self, frame_num: int) -> bool:
        """设置当前指针位置。

        由于VideoCapture接口有限，帧定位和跳转可能存在如下问题：
        1. 对于部分编码损坏的视频，set_to会耗时很长，并且后续会读取失败。
        2. 对于关键帧较为稀疏的输入，无法准确跳转到指定位置。
        
        Args:
            frame_num (int): 期望跳转位置

        Returns:
            bool: 是否成功跳转
        """
        return self.video.set(cv2.CAP_PROP_POS_MSEC,
                              frame2time(frame_num, self.fps))

    def force_set_to(self, frame_num: int) -> bool:
        """逐帧索引，直接跳转的降级方案。

        Args:
            frame_num (int): 期望跳转位置

        Returns:
            bool: 是否成功跳转
        """
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        status = True
        for _ in range(frame_num):
            status = self.video.grab()
            if not status: return status
        return status

    def get_video_pos(self):
        #return int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
        return time2frame(int(self.video.get(cv2.CAP_PROP_POS_MSEC)), self.fps)


class PyAVVideoWrapper(BaseVideoWrapper):
    """VideoWrapper for pyav-based video loader.

    Args:
        video_name (str): The video filename.

    Raises:
        av.error.FFmpegError could be raised during av.open.
    """

    def __init__(self, video_name: str) -> None:
        self.container = av.open(video_name,
                                 options={'threads': str(os.cpu_count())})
        self.video = self.container.streams.video[0]
        self.video.thread_type = "FRAME"

    @property
    def num_frames_by_container(self):
        # serve as a provision.
        if self.container.duration is None:
            return 0
        return int(self.container.duration / 1e6 * self.fps)

    @property
    def fps(self):
        # base_rate?
        return float(self.video.average_rate) if self.video.average_rate else 0

    @property
    def backend_name(self):
        return self.__class__.__name__ + "(FFmpeg)"

    @property
    def num_frames(self):
        return self.video.frames if self.video.frames != 0 else self.num_frames_by_container

    @property
    def size(self):
        return [int(self.video.width), int(self.video.height)]

    def read(self):
        try:
            while True:
                frame: list[av.VideoFrame] = self.container.demux(
                    video=0).__next__().decode()  # type: ignore
                if len(frame) == 0:
                    continue
                return True, frame[0].to_ndarray(format='bgr24')
        except av.error.EOFError:
            return False, None

    def release(self):
        self.container.close()

    def set_to(self, frame_num: int):
        """设置当前指针位置。
        """
        if self.video.time_base is None:
            raise av.error.ValueError(
                code=-1,
                message="Invalid time_base value: None",
            )
        # backward seeking makes sure cur frame is before the target.
        # seems seek using us instead of ms.
        self.container.seek(frame2time(frame_num, self.fps) * 1000,
                            any_frame=False,
                            backward=True)
        # 2-stage seeking, decoding until find the frame_num.
        for packet in self.container.demux(video=0):
            for decoded_frame in packet.decode():
                cur_frame = self.pts2frame(decoded_frame.pts)
                if cur_frame >= frame_num:
                    return True
        return True

    def force_set_to(self, frame_num: int) -> bool:
        self.container.seek(0, any_frame=False, backward=True)
        # demux without decoding to fast seek
        for packet in self.container.demux(video=0):
            for decoded_frame in packet.decode():
                cur_frame = self.pts2frame(decoded_frame.pts)
                if cur_frame >= frame_num:
                    return True
        return True

    def get_video_pos(self) -> int:
        while True:
            frame = self.container.demux(video=0).__next__().decode()
            if len(frame) == 0:
                continue
            return self.pts2frame(frame[0].pts)

    def pts2frame(self, pts: int):
        if self.video.time_base is None:
            return -1
        return int(pts * float(self.video.time_base) * self.fps)

    def frame2pts(self, frame_num: int):
        if self.video.time_base is None:
            return -1
        return int(frame_num / self.fps / self.video.time_base)
