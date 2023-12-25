"""
VideoLoader handles the whole video-to-frame process, which includes: 
    1. mask loading; 
    2. running-time video resolution option parsing; 
    3. preprocessing pipeline building; 
    4. exposure time estimation.
    5. get processed image.

VideoLoader 提供了从原始视频到处理后帧图像的流程控制，主要包含：
    1. 加载掩模；
    2. 解析尺寸选项；
    3. 构建预处理管线；
    4. 估算曝光时间；
    5. 获取处理后图像.
    
"""

import queue
import threading
from abc import ABCMeta, abstractmethod
from math import floor
from typing import Optional, Type, Union, Any

import numpy as np

from .MetLog import get_default_logger
from .utils import (MergeFunction, Transform, load_8bit_image,
                    parse_resize_param, timestr2int, transpose_wh, time2frame,
                    frame2time, sigma_clip)
from .VideoWarpper import BaseVideoWarpper

UP_EXPOSURE_BOUND = 0.5
DEFAULT_EXPOSURE_FRAME = 1
SHORT_LENGTH_THRESHOLD = 300
RF_ESTIMATE_LENGTH = 100
SLOW_EXP_TIME = 1 / 4


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
        start_time (int): the time (in ms) that VideoLoader starts reading.
        end_time (int): the time (in ms) that VideoLoader ends reading.
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
    2. Pop 1 frame from its queue with the .pop() method.
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
    2. Pop 1 frame with the .pop() method.
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
        start_frame = time2frame(timestr2int(start_time),
                                 self.fps) if start_time != None else 0
        end_frame = time2frame(
            timestr2int(end_time),
            self.fps) if end_time != None else self.video_total_frames
        self.reset(start_frame, end_frame, exp_frame=DEFAULT_EXPOSURE_FRAME)

        # construct merge function
        self.merge_func: Any = getattr(MergeFunction, merge_func, None)
        assert callable(self.merge_func), NameError(
            f"Unsupported merge function name: {merge_func}.")

        # Generate preprocessing function
        # Resize, Mask are Must-to-do things, while grayscale is selective.
        preprocess = []
        if self.raw_size != self.runtime_size:
            preprocess.append(
                Transform.opencv_resize(self.runtime_size, **kwargs))
        if self.grayscale:
            preprocess.append(Transform.opencv_BGR2GRAY())
        if self.mask_name:
            preprocess.append(Transform.mask_with(self.mask))
        self.preprocess = Transform.compose(preprocess)

        # init exposure time (exp_time) and exposure frame (exp_frame)
        self.exp_time = self.init_exp_time(exp_option,
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
            if self.grayscale:
                return np.ones(transpose_wh(self.runtime_size), dtype=np.uint8)
            else:
                return np.ones(transpose_wh(self.runtime_size+[3]), dtype=np.uint8)
        mask = load_8bit_image(mask_fname)
        mask_transforms = [Transform.opencv_resize(self.runtime_size)]
        if mask_fname.lower().endswith(".jpg"):
            mask_transforms.append(Transform.opencv_BGR2GRAY())
            mask_transforms.append(Transform.opencv_binary(128, 1))
        elif mask_fname.lower().endswith(".png"):
            # 对于png，仅取透明度层，且逻辑取反
            mask = mask[:, :, -1]
            mask_transforms.append(Transform.opencv_binary(128, 1, inv=True))

        if not self.grayscale:
           mask_transforms.append(Transform.expand_3rd_channel(3))

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
        """set `start_frame`, `end_frame`, and `exp_frame` of VideoLoader.
        
        Notice: `reset` is a lazy method. The start position is truly reset when the `start` method is called.

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
            f"Preset start_frame to {self.start_frame}; end_frame to {self.end_frame}."
        )

    def pop(self) -> np.ndarray:
        """pop a frame that can be used for detection.

        Returns:
            np.ndarray: processed frame.
        """
        frame_list = []
        for _ in range(self.exp_frame):
            status, self.cur_frame = self.video.read()
            if status:
                frame_list.append(self.preprocess(self.cur_frame))
            else:
                self.stop()
                break
        self.cur_iter -= self.exp_frame
        if self.cur_iter <= 0: self.stop()

        if self.exp_frame == 1:
            return frame_list[0]
        return self.merge_func(frame_list)

    def stop(self):
        self.logger.debug("Video stop triggered.")
        self.read_stopped = True

    def release(self):
        """The `.stop()` function will be called before `.release()`.
        """
        if not self.stopped:
            self.stop()
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
            (f" Mask path: \"{self.mask_name}\";" if self.mask_name else " Mask: None")+ "\n" +\
            f"    Video frames = {self.video_total_frames}; Apply grayscale = {self.grayscale};\n"+\
            f"    Raw resolution = {self.raw_size}; Running-time resolution = {self.runtime_size};\n"+\
            f"Apply exposure time of {self.exp_time:.2f}s."+\
            f"(MinTimeFlag = {frame2time(self.exp_frame * self.eq_int_fps, self.fps)})\n" +\
            f"Total frames = {self.iterations} ; FPS = {self.fps:.2f} (rFPS = {self.eq_fps:.2f})"

    def init_exp_time(self, exp_time: Union[int, float, str],
                      upper_bound: float) -> float:
        """Init exposure time. Return the exposure time.

        Args:
            exp_time (int,float,str): value from config.json. It can be either a value or a specified string.
            mask (np.array): mask array.

        Raises:
            ValueError: raised if the exp_time is invalid.

        Returns:
            exp_time: the exposure time in float.
        """
        # TODO: Rewrite this annotation.
        # TODO 2: set `upbound` to json setting.
        self.logger.info(f"Parsing \"exp_time\"={exp_time}")
        fps = self.video.fps
        self.logger.info(f"Metainfo FPS = {fps:.2f}")
        assert isinstance(
            exp_time, (str, float, int)
        ), f"exp_time should be either <str, float, int>, got {type(exp_time)}."

        if fps <= int(1 / upper_bound):
            self.logger.warning(
                f"Slow FPS detected. Use {1/fps:.2f}s directly.")
            return 1 / fps

        if isinstance(exp_time, str):
            if exp_time == "real-time":
                return 1 / fps
            if exp_time == "slow":
                return SLOW_EXP_TIME
            if exp_time == "auto":
                rf = rf_estimator(self)
                if rf / fps >= upper_bound:
                    self.logger.warning(
                        f"Unexpected exposuring time (too long):{rf/fps:.2f}s. Use {upper_bound:.2f}s instead."
                    )
                return min(rf / fps, upper_bound)
            try:
                exp_time = float(exp_time)
            except ValueError as E:
                raise ValueError(
                    "Invalid exp_time string value: It should be selected from [float], [int], "
                    + "real-time\",\"auto\" and \"slow\", got %s." %
                    (exp_time))
        if isinstance(exp_time, (float, int)):
            if exp_time * fps < 1:
                self.logger.warning(
                    f"Invalid exposuring time (too short). Use {1/fps:.2f}s instead."
                )
                return 1 / fps
            return float(exp_time)
        return 0


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
    2. Pop 1 frame from its queue with the .pop() method. 
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
        self.maxsize = maxsize
        self.queue = queue.Queue(maxsize=self.maxsize)
        super().__init__(video_warpper, video_name, mask_name, resize_option,
                         start_time, end_time, grayscale, exp_option,
                         merge_func, **kwargs)

    def clear_queue(self):
        """clear queue.
        """
        while not self.queue.empty():
            self.queue.get()

    def start(self):
        self.clear_queue()
        self.read_stopped = False
        self.status = True
        # TODO: 对于部分编码损坏的视频，set_to会耗时很长，并且后续会读取失败。应当做对应处置。
        # TODO 2: 对于不能set_to的，能否向后继续跳转？
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
        try:
            for i in range(self.exp_frame):
                if self.stopped: break
                ret.append(self.queue.get(timeout=2))
        except Exception as e:
            # handle the condition when there is no frame to read due to manual stop trigger or other exception.
            if isinstance(e,queue.Empty) and self.read_stopped:
                self.logger.info("Acceptable queue.Empty exception occured.")
                pass
            else:
                raise e
        if len(ret)==0:
            return None
        return self.merge_func(ret)

    def load_a_frame(self):
        """Load a frame from the video object.

        Returns:
            bool : status code. 1 for success operation, 0 for failure.
        """
        self.status, self.cur_frame = self.video.read()
        if self.status:
            self.processed_frame = self.preprocess(self.cur_frame)
            self.queue.put(self.processed_frame, timeout=10)
            return True
        else:
            self.stop()
            return False

    def videoloop(self):
        try:
            for i in range(self.iterations):
                if self.read_stopped or not self.status:
                    break
                if not self.load_a_frame():
                    self.logger.warning(f"Load frame failed at {self.start_frame + i}")
                    break
        except Exception as e:
            raise e
        finally:
            self.stop()

    def stop(self):
        if not self.read_stopped:
            super().stop()

    def release(self):
        super().release()
        if self.queue.not_empty:
            self.clear_queue()

    @property
    def stopped(self) -> bool:
        return self.read_stopped and self.queue.empty()

    def is_empty(self):
        return self.queue.empty()


def _rf_est_kernel(video_loader):
    try:
        n_frames = video_loader.iterations
        video_loader.start()
        f_sum = np.zeros((n_frames, ), dtype=float)
        for i in range(n_frames):
            if not video_loader.stopped:
                frame = video_loader.pop()
                f_sum[i] = np.sum(frame)
            else:
                f_sum = f_sum[:i]
                break

        A0, A1, A2, A3 = f_sum[:-3], f_sum[1:-2], f_sum[2:-1], f_sum[3:]

        diff_series = f_sum[1:] - f_sum[:-1]
        rmax_pos = np.where((2 * A2 - (A1 + A3) > 0) & (2 * A1 - (A0 + A2) < 0)
                            & (np.abs(diff_series[1:-1]) > 0.01))[0]
        #plt.scatter(rmax_pos + 1, diff_series[rmax_pos + 1], s=30, c='r')
        #plt.plot(diff_series, 'r')
        #plt.show()
    finally:
        video_loader.stop()
    return rmax_pos[1:] - rmax_pos[:-1]


def rf_estimator(video_loader)-> Union[float, int]:
    """用于为给定的视频估算实际的曝光时间。

    部分相机在录制给定帧率的视频时，可以选择慢于帧率的单帧曝光时间（慢门）。
    还原真实的单帧曝光时间可帮助更好的检测。
    但目前没有做到很好的估计。

    Args:
        video_loader (BaseVideoLoader): 待确定曝光时间的VideoLoader。
        mask (ndarray): the mask for the video.
    """
    start_frame, end_frame = video_loader.start_frame, video_loader.end_frame
    iteration_frames = video_loader.iterations

    # 估算时，将强制设置exp_frame=1以进行估算
    raw_exp_frame = video_loader.exp_frame
    video_loader.exp_frame = 1

    if iteration_frames < SHORT_LENGTH_THRESHOLD:
        # 若不超过300帧 则进行全局估算
        intervals = _rf_est_kernel(video_loader)
    else:
        # 超过300帧的 从开头 中间 结尾各抽取100帧长度的视频进行估算。
        video_loader.reset(end_frame=start_frame + RF_ESTIMATE_LENGTH, )
        intervals_1 = _rf_est_kernel(video_loader)

        video_loader.reset(start_frame=start_frame +
                           (iteration_frames - RF_ESTIMATE_LENGTH) // 2,
                           end_frame=start_frame +
                           (iteration_frames + RF_ESTIMATE_LENGTH) // 2)
        intervals_2 = _rf_est_kernel(video_loader)

        video_loader.reset(start_frame=end_frame - RF_ESTIMATE_LENGTH,
                           end_frame=end_frame)
        intervals_3 = _rf_est_kernel(video_loader)
        intervals = np.concatenate([intervals_1, intervals_2, intervals_3])

    # 还原video_reader的相关设置
    video_loader.exp_frame = raw_exp_frame
    video_loader.reset(start_frame, end_frame)

    if len(intervals) == 0:
        return 1

    # 非常经验的取值方法...
    est_frames: np.floating = np.round(
        np.min([np.median(intervals),
                np.mean(sigma_clip(intervals))]))
    return est_frames # type: ignore
