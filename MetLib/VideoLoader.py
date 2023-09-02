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


class BaseVideoLoader(metaclass=ABCMeta):
    """ 
    # BaseVideoLoader

    BaseVideoLoader is an abstract base class of VideoLoader. You can inherit BaseVideoLoader to implement your VideoLoader.

    VideoLoader class handles the whole video-to-frame process, which should include: 
    1. mask loading; 
    2. running-time video resolution option parsing; 
    3. preprocessing pipeline building; 
    4. exposure time estimation.

    notice: It is advised to use a VideoWarpper to load video,\
        since all VideoWarpper(s) share the same basic APIs \
        (so that it is easy to support other kinds of video sources by replacing VideoWarpper).

    ## What VideoLoader should support
    ### Property
    #### video attributes
        video_total_frames ([int]): num of total frames of the video.
        raw_size (Union[list, tuple]): the raw image/video resolution, in [w, h] order.
        runntime_size (Union[list, tuple]): the running image/video resolution, in [w, h] order.
        fps (float): fps of the video. 
    
    #### reading attributes
        start_time (int): the time (in ms) that VideoReader starts reading.
        end_time (int): the time (in ms) that VideoReader ends reading.
        start_frame (int): the start frame that is corresponding to the start_time.
        end_frame (int): the end frame that is corresponding to the end_time.
        iterations (int): the number of frames that is to be read.

    #### equivalent exposure time and equivalent fps
        exp_time (float): calculated exposure time (or is set through exp_option).
        exp_frame (int): the number of continuous frames that can be considered as the same frame that lasts.
        eq_fps (float): equivalent fps (based on exp_time instead of fps from metadata).
        eq_int_fps (int): integer of eq_fps.
    
    #### others
        mask (np.ndarray): the mask used in reading.
        cur_frame (np.ndarray): The last frame being read out.
        stopped (bool): return True if all frames are read, otherwise False.

    #### Method
        start(): start video loading.
        stop(): end video loading.
        reset(start_frame=None, end_frame=None): reset reading status of VideoLoader(include the start frame, \
                                                the end frame, the num of exposure frames, etc.)
        pop(): return a processed frame that can be directly used for detection.
        release(): release the video. Triggered when the program finishes.
        summary(): return the basic information about the VideoLoader.
    
    ## Usage

    All of VideoLoader (take T=VideoLoader(args,kwargs) as an example) classes should be designed and 
    utilized following these instructions:
    
    1. Call .start() method before using it. eg. : T.start()
    2. Pop 1 frame from its frame_pool with the .pop() method.
    3. when its video reaches the EOF or an exception is raised, its .stop() method should be triggered. 
       Then T.stopped will be set to True to ensure other parts of the program be terminated normally.
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


class VanillaVideoLoader(BaseVideoLoader):
    """ 
    # VanillaVideoLoader
    VanillaVideoLoader is a basic implementation that loads the video from the file.

    In this implementation, the video is only read when the pop method is called,\
        which can suffer from file IO blocking.
    ## Args:
        video_warpper (Type[BaseVideoWarpper]): the type of videowarpper.
        video_name (str): the filename of the video.
        mask_name (Optional[str]): the filename of the mask. The default is None.
        resize_option (Union[int, list, str, None]): resize option from input. The default is None.
        start_time (Optional[str]): the start time string of the video (like "HH:MM:SS" or "6000"(ms)). The default is None.
        end_time (Optional[str]): the start time string of the video (like "HH:MM:SS" or "6000"(ms)). The default is None.
        grayscale (bool): whether to use the grayscale image to accelerate calculation. The default is False.
        exp_option (Union[int, float, str]): resize option from input. The default is "auto".
        merge_func (str): the name of the preprocessing function that merges several frames into one frame. The default is "not_merge".
        **kwargs: compatibility design to support other arguments. 
            VanilliaVideoLoader support: dict(resize_interpolation=[opencv_intepolation_option])

    ## Usage

    VanillaVideoLoader can be utilized following these instructions:
    
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
        # VanillaVideoLoader
        VanillaVideoLoader is a basic implementation that loads the video from the file.

        In this implementation, the video is only read when the pop method is called,\
            which can suffer from file IO blocking.

        ## Args:
            video_warpper (Type[BaseVideoWarpper]): the type of videowarpper.
            video_name (str): the filename of the video.
            mask_name (Optional[str]): the filename of the mask. The default is None.
            resize_option (Union[int, list, str, None]): resize option from input. The default is None.
            start_time (Optional[str]): the start time string of the video (like "HH:MM:SS" or "6000"(ms)). The default is None.
            end_time (Optional[str]): the start time string of the video (like "HH:MM:SS" or "6000"(ms)). The default is None.
            grayscale (bool): whether to use the grayscale image to accelerate calculation. The default is False.
            exp_option (Union[int, float, str]): resize option from input. The default is "auto".
            merge_func (str): the name of the preprocessing function that merges several frames into one frame. The default is "not_merge".
            **kwargs: compatibility design to support other arguments. 
                VanilliaVideoLoader support: dict(resize_interpolation=[opencv_intepolation_option])

        ## Raises:
            NameError: Raised when asking for a undefined merge function.
        """

        # init necessary variables
        self.video_name = video_name
        self.mask_name = mask_name
        self.grayscale = grayscale
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

        # init exposure time (exp_time) and exposure frame (exp_frame)
        self.exp_time = init_exp_time(exp_option,
                                      self,
                                      upper_bound=UP_EXPOSURE_BOUND)
        self.exp_frame = int(round(self.exp_time * self.fps))

        assert not (
            self.merge_func == MergeFunction.not_merge and self.exp_frame != 1
        ), "Cannot \"not_merge\" frames when num of exposure frames > 1. Please specify a merge function."

    def load_mask(self, mask_fname: Union[str, None]) -> np.ndarray:
        """Load mask from the given path `mask_fname` and rescale it.

        Args:
            mask_fname (str): path to the mask.

        Returns:
            np.ndarray: the resized mask.
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
             Transform.expand_3rd_channel(1)])

        return Transform.compose(mask_transforms)(mask)

    def start(self):
        self.cur_iter = self.iterations
        self.read_stopped = False
        self.video.set_to(self.start_frame)

    def reset(self,
              start_frame: Union[int, None] = None,
              end_frame: Union[int, None] = None,
              exp_frame: Union[int, None] = None,
              reset_time_attr: bool = True):
        """set `start_frame`, `end_frame`, and `exp_frame` of VideoReader.
        
        Notice: `reset` is a lazy method. The start position is reset when the `start` method is called.

        Args:
            start_frame (Union[int, None], optional): the start frame of the video. Defaults to None.
            end_frame (Union[int, None], optional): the end frame of the video. Defaults to None.
            exp_frame (Union[int, None], optional): the exposure frame of the video. Defaults to None.
            reset_time_attr (bool, optional): whether to reset time attributes of VideoLoader. Defaults to True.
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

    def pop(self) -> np.ndarray:
        """pop a frame that can be used for detection.

        Returns:
            np.ndarray: processed frame.
        """
        frame_pool = []
        for _ in range(self.exp_frame):
            status, self.cur_frame = self.video.read()
            if status:
                frame_pool.append(self.preprocess(self.cur_frame))
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
    def video_total_frames(self) -> int:
        return self.video.num_frames

    @property
    def raw_size(self) -> Union[list, tuple]:
        return self.video.size

    @property
    def eq_fps(self) -> float:
        return 1 / self.exp_time

    @property
    def eq_int_fps(self) -> int:
        return floor(self.eq_fps)

    def summary(self) -> str:
        return f"{self.__class__.__name__} summary:\n"+\
            f"    Video path: \"{self.video_name}\";"+\
            (f" Mask path: \"{self.mask_name}\";" if self.mask_name else "Mask: None")+ "\n" +\
            f"    Video frames = {self.video_total_frames}; Apply grayscale = {self.grayscale};\n"+\
            f"    Raw resolution = {self.raw_size}; Running-time resolution = {self.runtime_size};\n"+\
            f"Apply exposure time of {self.exp_time:.2f}s."+\
            f"(MinTimeFlag = {1000 * self.exp_frame * self.eq_int_fps / self.fps})\n" +\
            f"Total frames = {self.iterations} ; FPS = {self.fps:.2f} (rFPS = {self.eq_fps:.2f})"


class ThreadVideoLoader(VanillaVideoLoader):
    """ 
    # ThreadVideoLoader
    This class is used to load the video from the file with an independent subthread.  
    ThreadVideoLoader can partly solve I/O blocking and provide speedup.

    ## Args:
        video_warpper (Type[BaseVideoWarpper]): the type of videowarpper.
        video_name (str): the filename of the video.
        mask_name (Optional[str]): the filename of the mask. The default is None.
        resize_option (Union[int, list, str, None]): resize option from input. The default is None.
        start_time (Optional[str]): the start time string of the video (like "HH:MM:SS" or "6000"(ms)). The default is None.
        end_time (Optional[str]): the start time string of the video (like "HH:MM:SS" or "6000"(ms)). The default is None.
        grayscale (bool): whether to use the grayscale image to accelerate calculation. The default is False.
        exp_option (Union[int, float, str]): resize option from input. The default is "auto".
        merge_func (str): the name of the preprocessing function that merges several frames into one frame. The default is "not_merge".
        maxsize (int): the maxsize of the video buffer queue. The default is 32.
        **kwargs: compatibility design to support other arguments. 
            ThreadVideoLoader support: dict(resize_interpolation=[opencv_intepolation_option])
    
    ## Usage

    All of VideoLoader (take T=VideoLoader(args,kwargs) as an example) classes should be designed and 
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
        # Queue is used in ThreadVideoReader to manage frame_pool
        # TODO: maybe I should change a name...?
        self.maxsize = maxsize
        self.frame_pool = queue.Queue(maxsize=self.maxsize)
        super().__init__(video_warpper, video_name, mask_name, resize_option,
                         start_time, end_time, grayscale, exp_option,
                         merge_func, **kwargs)

    def clear_frame_pool(self):
        """clear queue.
        """
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
            # this is abnormal. so the video file will be released manuly here.
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
        return self.read_stopped and self.frame_pool.empty()

    def is_empty(self):
        return self.frame_pool.empty()