import queue
import threading
import time
from abc import ABCMeta, abstractmethod
from math import floor
from typing import Optional, Type, Union

import numpy as np

from .MetLog import get_default_logger
from .utils import (MergeFunction, Transform, init_exp_time, load_8bit_image,
                    parse_resize_param, timestr2int, transpose_wh)
from .VideoWarpper import BaseVideoWarpper

UP_EXPOSURE_BOUND = 0.5
DEFAULT_EXPOSURE_FRAME = 1


class BaseVideoReader(metaclass=ABCMeta):
    """TODO: Add note for this.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def reset(self, start_frame=None, end_frame=None):
        pass

    @abstractmethod
    def pop(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def release(self):
        pass

    @property
    @abstractmethod
    def fps(self):
        pass

    @property
    @abstractmethod
    def video_total_frames(self):
        pass

    @property
    @abstractmethod
    def raw_size(self):
        pass


class VanillaVideoReader(BaseVideoReader):
    """ 
    # VanillaVideoReader
    This class is used to load the video from the file.

    In this basic implementation, video are loaded every time .pop() method is called, 
    which is an block file IO implementation.

    ## Args:
        video_warpper (BaseVideoWarpper): The type of videowarpper.
        video_name (str): The filename of the video.
        start_frame (int): The start frame of the video.
        iterations (int): The total number of frames that are going to load.
        preprocess (Callable): the preprocessing function that only takes frames[ndarray] as the only argument. 
                            You can use functools.partical to construct such a function.
        exp_frame (int): _description_
        merge_func (Callable): the preprocessing function that merges several frames to one frame. Take only 1 argument. 
                            You can use functools.partical to construct such a function.
    
    ## Usage

    All of VideoReader (take T=VideoReader(args,kwargs) as an example) classes should be designed and 
    utilized following these instructions:
    
    1. Call .start() method before using it. eg. : T.start()
    2. Pop 1 frame from its frame_pool with the .pop() method.
    3. when its video reaches the EOF or an exception is raised, its .stop() method should be triggered. 
       Then T.stopped will be set to True to ensure other parts of the program be terminated normally.
    """

    def __init__(self,
                 video_warpper: Type[BaseVideoWarpper],
                 video_name: str,
                 mask_name: Optional[str] = None,
                 resize_option: Union[int, list, str, None] = None,
                 start_time: Optional[str] = None,
                 end_time: Optional[str] = None,
                 grayscale: bool = False,
                 exp_option: Union[int, float, str] = "auto",
                 merge_func: str = "not_merge",
                 **kwargs) -> None:
        """
        VanillaVideoReader is a basic class that can be used to load a video.

        ## Args:
            video_warpper (BaseVideoWarpper): The type of videowarpper.
            video_name (str): The filename of the video.
            start_frame (int): The start frame of the video.
            iterations (int): The total number of frames that are going to load.
            exp_frame (int): _description_
            merge_func (Callable): the preprocessing function that merges several frames to one frame. Take only 1 argument. 
                             You can use functools.partical to construct such a function.

        ## Raises:
            NameError: Raised when asking for a undefined merge function.
        """
        self.video_name = video_name
        self.mask_name = mask_name
        self.grayscale = grayscale

        # init necessary variables
        self.logger = get_default_logger()
        self.status = True
        self.read_stopped = True

        # load video and mask
        self.video = video_warpper(video_name)
        self.runtime_size = parse_resize_param(resize_option, self.raw_size)
        self.mask = self.load_mask(self.mask_name)

        # init reader status (start and end time)
        self.start_time = timestr2int(start_time) if start_time != None else 0
        self.end_time = timestr2int(end_time) if end_time != None else (
            self.video_total_frames / self.fps * 1000)
        start_frame = int(self.start_time / 1000 * self.fps)
        end_frame = int(self.end_time / 1000 * self.fps)
        self.reset(start_frame, end_frame, exp_frame=DEFAULT_EXPOSURE_FRAME)

        # construct merge function
        self.merge_func = getattr(MergeFunction, merge_func, None)
        assert self.merge_func is not None, NameError(
            f"Unsupported merge function name: {merge_func}.")

        # Generate preprocessing function
        # Resize, Mask are Must-to-do things, while grayscale is selective.
        preprocess = []
        if self.raw_size != self.runtime_size:
            preprocess.append(
                Transform.opencv_resize(self.runtime_size, **kwargs))
        if self.grayscale:
            preprocess.append(Transform.opencv_BGR2GRAY())
        preprocess.append(Transform.expand_3rd_channel(1))
        if self.mask_name:
            preprocess.append(Transform.mask_with(self.mask))  # type: ignore
        self.preprocess = Transform.compose(preprocess)
        # 计算曝光时间 & 曝光帧数
        self.exp_time = init_exp_time(exp_option,
                                      self,
                                      upper_bound=UP_EXPOSURE_BOUND)
        self.exp_frame = int(round(self.exp_time * self.fps))

        assert not (
            self.merge_func == MergeFunction.not_merge and self.exp_frame != 1
        ), "Cannot \"not_merge\" frames when num of exposure frames > 1. Please specify a merge function."

    def load_mask(self, mask_fname: Union[str, None]):
        """从给定路径加载mask，并根据video尺寸及是否单色(grayscale)转换mask。

        Args:
            mask_fname (str): mask路径

        Returns:
            _type_: _description_
        """
        if mask_fname == None:
            return np.ones(transpose_wh(self.runtime_size), dtype=np.uint8)
        mask = load_8bit_image(mask_fname)
        mask_transforms = [Transform.opencv_resize(self.runtime_size)]
        if mask_fname.lower().endswith(".jpg"):
            mask_transforms.append(Transform.opencv_BGR2GRAY())
        elif mask_fname.lower().endswith(".png"):
            mask = mask[:, :, -1:]
        mask_transforms.extend(
            [Transform.opencv_binary(128, 1),
             Transform.expand_3rd_channel(1)])  # type: ignore

        return Transform.compose(mask_transforms)(mask)

    def start(self):
        self.cur_iter = self.iterations
        self.read_stopped = False
        self.video.set_to(self.start_frame)

    # TODO: 名字也改一下
    def reset(self,
              start_frame: Union[int, None] = None,
              end_frame: Union[int, None] = None,
              exp_frame: Union[int, None] = None,
              reset_time_attr: bool = False):
        """设置VideoLoader的起始，结束时间帧及单次曝光持续帧数。
        
        需要注意的是，仅在VideoReader.start()之后才真正对输入视频设置位置。

        Args:
            frame (_type_): _description_
        """
        assert self.read_stopped, f"Cannot reset a running {self.__class__.__name__}."

        if start_frame != None:
            self.start_frame = max(0, start_frame)
        if end_frame != None:
            self.end_frame = min(end_frame, self.video_total_frames)

        assert 0 <= self.start_frame < self.end_frame, ValueError(
            "Invalid start time or end time.")

        if exp_frame != None:
            self.exp_frame = exp_frame
        if reset_time_attr:
            self.start_time = self.start_frame / self.fps
            self.end_time = self.end_frame / self.fps
        self.iterations = self.end_frame - self.start_frame
        self.read_stopped = True

        self.logger.debug(
            f"set start_frame to {self.start_frame}; end_frame to {self.end_frame}."
        )

    def pop(self):
        frame_pool = []
        for _ in range(self.exp_frame):
            status, frame = self.video.read()
            if status:
                frame_pool.append(self.preprocess(frame))
            else:
                self.stop()
                break
        self.cur_iter -= self.exp_frame
        if self.cur_iter <= 0: self.stop()

        if self.exp_frame == 1:
            return frame_pool[0]
        return self.merge_func(frame_pool)

    def stop(self):
        self.logger.debug("Video stop triggered.")
        # 原始实现是非异步的，因此stopped触发时必定已经结束
        self.read_stopped = True

    def release(self):
        self.video.release()

    @property
    def stopped(self) -> bool:
        return self.read_stopped

    @property
    def fps(self) -> float:
        return self.video.fps

    @property
    def video_total_frames(self):
        return self.video.num_frames

    @property
    def raw_size(self):
        """The size of the input video in [w, h] format.
        """
        return self.video.size

    @property
    def eq_fps(self):
        return 1 / self.exp_time

    @property
    def eq_int_fps(self):
        return floor(self.eq_fps)

    def summary(self):
        return f"{self.__class__.__name__} summary:\n"+\
            f"    Video path: \"{self.video_name}\";"+\
            (f" Mask path: \"{self.mask_name}\";" if self.mask_name else "Mask: None")+ "\n" +\
            f"    Video frames = {self.video_total_frames}; Apply grayscale = {self.grayscale};\n"+\
            f"    Raw resolution = {self.raw_size}; Running-time resolution = {self.runtime_size};\n"+\
            f"Apply exposure time of {self.exp_time:.2f}s."+\
            f"(MinTimeFlag = {1000 * self.exp_frame * self.eq_int_fps / self.fps})\n" +\
            f"Total frames = {self.iterations} ; FPS = {self.fps:.2f} (rFPS = {self.eq_fps:.2f})"


class ThreadVideoReader(VanillaVideoReader):
    """ 
    # ThreadVideoReader
    This class is used to load the video from the file with an independent subthread.  
    ThreadVideoReader can partly solve I/O blocking and provide speedup.

    ## Args:

        video (Any): The video object that supports .read() method to load the next frame. 
                    We recommend to use cv2.VideoCapture object.
        iterations (int): The number of frames that are going to load.
        preprocess (func): the preprocessing function that only takes frames[ndarray] 
                    as the only arguments. You can use functools.partical to 
                    construct such a function.
        maxsize (int, optional): the max size of the frame buffer. Defaults to 30.
    
    ## Usage

    All of VideoReader (take T=VideoReader(args,kwargs) as an example) classes should be designed and 
    utilized following these instructions:
    
    1. Call .start() method before using it. eg. : T.start()
    2. Pop 1 frame from its frame_pool with the .pop() method. 
    3. when its video reaches the EOF or an exception is raised, its .stop() method should be triggered. 
       Then T.stopped will be set to True to ensure other parts of the program be terminated normally.
    """

    def __init__(self,
                 video_warpper: Type[BaseVideoWarpper],
                 video_name: str,
                 mask_name: Optional[str] = None,
                 resize_option: Union[int, list, str, None] = None,
                 start_time: Optional[str] = None,
                 end_time: Optional[str] = None,
                 grayscale: bool = False,
                 exp_option: Union[int, float, str] = "auto",
                 merge_func: str = "not_merge",
                 maxsize: int = 32,
                 **kwargs) -> None:
        # 对于ThreadVideoReader，使用Queue管理
        self.maxsize = maxsize
        self.frame_pool = queue.Queue(maxsize=self.maxsize)
        super().__init__(video_warpper, video_name, mask_name, resize_option,
                         start_time, end_time, grayscale, exp_option,
                         merge_func)

    def clear_frame_pool(self):
        while not self.frame_pool.empty():
            self.frame_pool.get()

    def start(self):
        self.clear_frame_pool()
        self.read_stopped = False
        self.status = True
        self.video.set_to(self.start_frame)
        self.thread = threading.Thread(target=self.videoloop, args=())
        self.thread.setDaemon(True)
        self.thread.start()
        return self

    def pop(self):
        if self.stopped:
            # this is abnormal. so the video file will be released here.
            self.video.release()
            self.thread.join()
            raise Exception(
                f"Attempt to read frame(s) from an ended {self.__class__.__name__} object."
            )
        ret = []
        for i in range(self.exp_frame):
            if self.stopped: break
            ret.append(self.frame_pool.get(timeout=2))
        return self.merge_func(ret)

    def load_a_frame(self):
        """Load a frame from the video object.

        Returns:
            bool : status code. 1 for success operation, 0 for failure.
        """
        self.status, self.cur_frame = self.video.read()
        if self.status:
            self.processed_frame = self.preprocess(self.cur_frame)
            self.frame_pool.put(self.processed_frame, timeout=2)
            return True
        else:
            self.stop()
            return False

    def videoloop(self):
        try:
            for i in range(self.iterations):
                if self.read_stopped or not self.status: break
                if not self.load_a_frame():
                    break
        finally:
            self.stop()

    def stop(self):
        if not self.read_stopped:
            super().stop()

    @property
    def stopped(self) -> bool:
        """当已经读取完毕时触发。
        """
        return self.read_stopped and self.frame_pool.empty()

    def is_empty(self):
        return self.frame_pool.empty()