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
from .metlog import get_default_logger

logger = get_default_logger()
PTS_JITTER_TOLERANCE_FRAMES = 0.5
PTS_GAP_WARNING_SEC = 0.5


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

    def __init__(self, video_name: str, hwaccel: Optional[str] = None) -> None:
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
        hwaccel (Optional[str]): The hardware acceleration type. 
            This is not actually working, just for compatibility.
    Raises:
        FileNotFoundError: triggered when the video file can not be opened. 
    """

    def __init__(self, video_name: str, hwaccel: Optional[str] = None) -> None:
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

    def __init__(self, video_name: str, hwaccel: Optional[str] = None) -> None:
        if hwaccel is not None:
            self.video_decoder = av.codec.hwaccel.HWAccel(
                device_type=hwaccel, allow_software_fallback=True)
        else:
            self.video_decoder = None
        self.container = av.open(video_name,
                                 options={'threads': str(os.cpu_count())},
                                 hwaccel=self.video_decoder)
        self.video = self.container.streams.video[0]
        self.video.thread_type = "FRAME"
        self.video_frame_cache: list[av.VideoFrame] = []
        # CFR output timeline. Input frames are positioned only by their PTS.
        self._target_fps = self._select_target_fps()
        self._num_frames = self._calc_output_frame_count()
        # 逻辑帧计数器，用于追踪实际帧位置
        self._cur_frame_idx = 0
        self._last_frame_data = None
        self._last_frame_time_sec: Optional[float] = None
        self._eof = False

    def _select_target_fps(self) -> float:
        """Select the nominal CFR used by downstream consumers.

        ``average_rate`` is affected by timestamp gaps and is therefore a bad
        choice for resampling. FFmpeg's guessed/base rate better describes the
        nominal cadence. ``average_rate`` remains the final compatibility
        fallback for unusual streams.
        """
        for rate_name in ("guessed_rate", "base_rate", "average_rate"):
            rate = getattr(self.video, rate_name, None)
            if rate is not None and float(rate) > 0:
                return float(rate)
        return 0.0

    def _stream_duration_sec(self) -> float:
        if self.video.duration is not None and self.video.time_base is not None:
            return float(self.video.duration * self.video.time_base)
        if self.container.duration is not None:
            return self.container.duration / 1e6
        return 0.0

    def _calc_output_frame_count(self) -> int:
        """Return the length of the CFR output timeline, including PTS gaps."""
        duration = self._stream_duration_sec()
        if duration > 0 and self._target_fps > 0:
            return int(round(duration * self._target_fps))
        return int(self.video.frames or 0)

    @property
    def num_frames_by_container(self):
        # serve as a provision.
        duration = self._stream_duration_sec()
        if duration <= 0 or self.fps <= 0:
            return 0
        return int(round(duration * self.fps))

    @property
    def fps(self):
        return self._target_fps

    @property
    def backend_name(self):
        return f"{self.__class__.__name__}(FFmpeg)({self.container.streams[0].codec_context.codec.name})"

    @property
    def num_frames(self):
        return self._num_frames

    @property
    def size(self):
        return [int(self.video.width), int(self.video.height)]

    def _frame_time_sec(self, frame: av.VideoFrame) -> Optional[float]:
        if frame.pts is None or self.video.time_base is None:
            return None
        start_pts = self.video.start_time or 0
        return float((frame.pts - start_pts) * self.video.time_base)

    @property
    def _jitter_tolerance_sec(self) -> float:
        if self.fps <= 0:
            return 0.0
        return PTS_JITTER_TOLERANCE_FRAMES / self.fps

    def _decode_next_batch(self) -> bool:
        if self.video_frame_cache:
            return True
        if self._eof:
            return False

        for frames in self._decoded_frame_batches():
            self.video_frame_cache.extend(frames)
            return True
        self._eof = True
        return False

    def _decoded_frame_batches(self):
        """Yield decoded frame batches and treat decoder flush EOF normally."""
        try:
            for packet in self.container.demux(self.video):
                frames: list[av.VideoFrame] = packet.decode()  # type: ignore
                if frames:
                    yield frames
        except av.error.EOFError:
            # Some codecs report the final decoder flush as AVERROR_EOF.
            return

    def _record_input_time(self, frame_time_sec: float) -> None:
        if self._last_frame_time_sec is not None:
            gap = frame_time_sec - self._last_frame_time_sec
            if gap > PTS_GAP_WARNING_SEC:
                logger.debug(
                    f"Video PTS gap detected: {gap:.3f}s between "
                    f"{self._last_frame_time_sec:.3f}s and "
                    f"{frame_time_sec:.3f}s.")
        self._last_frame_time_sec = frame_time_sec

    def _is_stale_frame_time(self, frame_time_sec: float,
                             reference_time_sec: Optional[float]) -> bool:
        return (reference_time_sec is not None and
                frame_time_sec < reference_time_sec -
                self._jitter_tolerance_sec)

    @staticmethod
    def _log_stale_frame(frame_time_sec: float,
                         reference_time_sec: float) -> None:
        logger.debug(
            f"Non-monotonic video PTS detected: {frame_time_sec:.3f}s is "
            f"behind {reference_time_sec:.3f}s; skipping the stale frame.")

    def _consume_until(
            self, target_time_sec: float) -> Optional[av.VideoFrame]:
        """Consume input frames through a CFR target time.

        The latest eligible frame is returned. The first frame beyond the
        target remains cached for the next output slot or post-seek read.
        """
        selected_frame: Optional[av.VideoFrame] = None

        while self._decode_next_batch():
            # 预读 next_time_sec 用于判断是否需要继续解码下一帧作为候选帧
            next_frame = self.video_frame_cache[0]
            next_time_sec = self._frame_time_sec(next_frame)

            # Streams without usable PTS fall back to sequential decoding.
            if next_time_sec is None:
                selected_frame = self.video_frame_cache.pop(0)
                break

            reference_time_sec = self._last_frame_time_sec
            if self._is_stale_frame_time(next_time_sec,
                                         reference_time_sec):
                self.video_frame_cache.pop(0)
                assert reference_time_sec is not None
                self._log_stale_frame(next_time_sec, reference_time_sec)
                continue

            if next_time_sec > target_time_sec + self._jitter_tolerance_sec:
                break

            selected_frame = self.video_frame_cache.pop(0)
            self._record_input_time(next_time_sec)

        return selected_frame

    def read(self):
        """Read one frame from the CFR output timeline.

        Input frames are selected by PTS. When the input is sparse or contains
        a timestamp gap, the latest frame is held. When multiple input frames
        fall into one output slot, only the latest one is returned.
        """
        try:
            if self._cur_frame_idx >= self.num_frames:
                return False, None
            if self.fps <= 0:
                return False, None

            output_time_sec = self._cur_frame_idx / self.fps
            selected_frame = self._consume_until(output_time_sec)

            if selected_frame is not None:
                self._last_frame_data = selected_frame.to_ndarray(format='bgr24')

            if self._last_frame_data is None:
                return False, None

            self._cur_frame_idx += 1
            return True, self._last_frame_data

        except Exception as e:
            logger.error(f"{e.__repr__()} encountered when reading "
                         f"video frame with {self.__class__.__name__}.")
            return False, None

    def release(self):
        self.container.close()

    def set_to(self, frame_num: int):
        """设置当前指针位置。
        """
        if self.video.time_base is None or self.fps <= 0:
            raise av.error.ValueError(
                code=-1,
                message="Invalid time_base value: None",
            )
        if frame_num < 0 or frame_num > self.num_frames:
            return False

        self._cur_frame_idx = frame_num
        self._last_frame_data = None
        self._last_frame_time_sec = None
        self.video_frame_cache = []
        self._eof = False

        if frame_num == self.num_frames:
            return True

        target_time_sec = frame_num / self.fps
        stream_start_sec = float(
            (self.video.start_time or 0) * self.video.time_base)
        self.container.seek(int(round((stream_start_sec + target_time_sec) * 1e6)),
                            any_frame=False,
                            backward=True)

        selected_frame = self._consume_until(target_time_sec)
        if selected_frame is not None:
            self._last_frame_data = selected_frame.to_ndarray(format='bgr24')

        if self.video_frame_cache and self._last_frame_time_sec is not None:
            next_time_sec = self._frame_time_sec(self.video_frame_cache[0])
            if (next_time_sec is not None and
                    next_time_sec - self._last_frame_time_sec >
                    PTS_GAP_WARNING_SEC):
                logger.debug(
                    f"Target {target_time_sec:.3f}s lies in a video PTS gap "
                    f"from {self._last_frame_time_sec:.3f}s to "
                    f"{next_time_sec:.3f}s; holding the previous frame.")

        return self._last_frame_data is not None

    def force_set_to(self, frame_num: int) -> bool:
        if not self.set_to(0):
            return False
        for _ in range(frame_num):
            status, _ = self.read()
            if not status:
                return False
        return True

    def get_video_pos(self) -> int:
        return self._cur_frame_idx

    def pts2frame(self, pts: int):
        if self.video.time_base is None or self.fps <= 0:
            return -1
        start_pts = self.video.start_time or 0
        return int(round((pts - start_pts) * float(self.video.time_base) *
                         self.fps))

    def frame2pts(self, frame_num: int):
        if self.video.time_base is None or self.fps <= 0:
            return -1
        start_pts = self.video.start_time or 0
        return start_pts + int(round(frame_num / self.fps /
                                     float(self.video.time_base)))
