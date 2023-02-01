import cv2
from abc import ABCMeta, abstractmethod

class BaseVideoWarpper(metaclass = ABCMeta):
    """VideoWarpper用于对加载视频的各类框架进行包装，
    对API进行统一，以使得可以简单的适用于各类VideoLoader。

    Args:
        object (_type_): _description_
    """

    def __init__(self) -> None:
        pass
    
    @property
    @abstractmethod
    def fps(self):
        pass

    @property
    @abstractmethod
    def frame_num(self):
        return int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    @abstractmethod
    def size(self):
        return (int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    @abstractmethod
    def set_to(self):
        pass
    
    def release(self):
        pass

    @abstractmethod
    def read(self):
        pass


class OpenCVVideoWarpper(object):
    """适用于OpenCV的VideoCapture所获得的Video。

    Args:
        object (_type_): _description_
    """

    def __init__(self, video_name) -> None:
        self.video = cv2.VideoCapture(video_name)
        if (self.video is None) or (not self.video.isOpened()):
            raise FileNotFoundError(
                f"The video \"{video_name}\" cannot be opened as a supported video format.")

    @property
    def fps(self):
        return self.video.get(cv2.CAP_PROP_FPS)

    @property
    def num_frames(self):
        return int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def size(self):
        return [int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))]

    def read(self):
        return self.video.read()

    def release(self):
        self.video.release()
    
    def set_to(self, frame):
        """设置当前指针位置。
        """
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame)