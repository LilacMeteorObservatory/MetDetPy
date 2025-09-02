from abc import ABCMeta
from typing import Any, Optional, Sequence, Union

import av
import cv2

from .metlog import BaseMetLog, get_useable_logger
from .utils import U8Mat, transpose_wh
from .videoloader import BaseVideoLoader

# TODO: save_video 和 save_video_by_stream 的底层通常一致，可以考虑做抽象。
# TODO: 通过 PyAV 支持编码输出和其他格式。


class BaseVideoWriter(ABCMeta):
    """
    VideoWriter a class that provides class methods about video writing.
    
    A video writer must support save a video from a Sequence(save_video)
    and save a video by stream(save_video_by_stream).
    """

    @classmethod
    def save_video(cls,
                   video_series: Sequence[U8Mat],
                   fps: Union[int, float],
                   video_path: str,
                   logger: Optional[BaseMetLog] = None,
                   *args: Any,
                   **kwargs: Any) -> int:
        raise NotImplementedError("...")

    @classmethod
    def save_video_by_stream(cls,
                             video_loader: BaseVideoLoader,
                             fps: Union[int, float],
                             video_path: str,
                             start_frame: Optional[int] = None,
                             end_frame: Optional[int] = None,
                             logger: Optional[BaseMetLog] = None) -> int:
        raise NotImplementedError("...")


class OpenCVVideoWriter(BaseVideoWriter):

    @classmethod
    def save_video(cls,
                   video_series: Sequence[U8Mat],
                   fps: Union[int, float],
                   video_path: str,
                   logger: Optional[BaseMetLog] = None,
                   *args: Any,
                   **kwargs: Any) -> int:
        """ Save a given video clip. Via OpenCV.
    
        Args:
            video_series (BaseVideoLoader): existing video series storage in np.ndarray or list format.
            fps (Union[int, float]): video fps.
            video_path (str): full output path of the video.
        """
        cv_writer = None
        logger = get_useable_logger(logger)
        try:
            real_size = transpose_wh(video_series[0].shape)[:2]
            cv_writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*"MJPG"),  # type: ignore
                fps,  # type: ignore
                real_size)
            for clip in video_series:
                cv_writer.write(clip)
        except Exception as e:
            logger.error(
                f"Failed to save video {video_path} because: {e.__repr__()}.")
            return -1
        finally:
            if cv_writer:
                cv_writer.release()
        return 0

    @classmethod
    def save_video_by_stream(cls,
                             video_loader: BaseVideoLoader,
                             fps: Union[int, float],
                             video_path: str,
                             start_frame: Optional[int] = None,
                             end_frame: Optional[int] = None,
                             logger: Optional[BaseMetLog] = None) -> int:
        """ Save video by stream(to avoid OutOfMemory). Via OpenCV.

        Args:
            video_loader (BaseVideoLoader): initialized video loader.
            fps (Union[int, float]): video fps.
            video_path (str): full output path of the video.
            start_frame (Optional[int], optional): the start frame of the stacker. Defaults to None.
            end_frame (Optional[int], optional): the end frame of the stacker.. Defaults to None.
            logger (Optional[Logger], optional): a logging.Logger object for logging use. Defaults to None.
        
        Return:
            int - return code. 0 for success.
        """
        logger = get_useable_logger(logger)
        if start_frame != None or end_frame != None:
            video_loader.reset(start_frame=start_frame, end_frame=end_frame)
        cv_writer = None
        try:
            video_loader.start()
            cv_writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*"MJPG"),  # type: ignore
                fps,
                video_loader.runtime_size)
            for _ in range(video_loader.iterations):
                cur_frame = video_loader.pop()
                if cur_frame is None: continue
                cv_writer.write(cur_frame)
        except Exception as e:
            if logger:
                logger.error(
                    f"Failed to save video {video_path} because: {e.__repr__()}."
                )
            return -1
        finally:
            video_loader.stop()
            if cv_writer:
                cv_writer.release()
        return 0


class PyAVVideoWriter(BaseVideoWriter):

    @classmethod
    def save_video(cls,
                   video_series: Sequence[U8Mat],
                   fps: Union[int, float],
                   video_path: str,
                   logger: Optional[BaseMetLog] = None,
                   *args: Any,
                   **kwargs: Any) -> int:
        """ Save a given video clip. Via OpenCV.
    
        Args:
            video_series (BaseVideoLoader): existing video series storage in np.ndarray or list format.
            fps (Union[int, float]): video fps.
            video_path (str): full output path of the video.
        """
        logger = get_useable_logger(logger)
        assert len(video_series) > 0, "Invalid video series!"
        sample_frame = video_series[0]
        (h, w, _) = sample_frame.shape
        try:
            # 创建容器并打开输出文件
            with av.open(video_path, mode="w") as av_writer:
                # 创建视频流，编码器使用 libx264，帧率为30
                stream = av_writer.add_stream(  # type: ignore
                    "libx264", rate=int(fps))
                stream.width = w
                stream.height = h
                stream.pix_fmt = "yuv420p"  # 通用的视频编码色彩空间

                for frame in video_series:
                    # 转换 numpy 数组为 PyAV 的 VideoFrame
                    frame = av.VideoFrame.from_ndarray(
                        frame, format="bgr24")  # type: ignore
                    for packet in stream.encode(frame):
                        av_writer.mux(packet)

                # flush编码器缓存
                for packet in stream.encode():
                    av_writer.mux(packet)
        except Exception as e:
            logger.error(
                f"Failed to save video {video_path} because: {e.__repr__()}.")
            return -1
        return 0

    @classmethod
    def save_video_by_stream(cls,
                             video_loader: BaseVideoLoader,
                             fps: Union[int, float],
                             video_path: str,
                             start_frame: Optional[int] = None,
                             end_frame: Optional[int] = None,
                             logger: Optional[BaseMetLog] = None) -> int:
        """ Save video by stream(to avoid OutOfMemory). 

        Args:
            video_loader (BaseVideoLoader): initialized video loader.
            fps (Union[int, float]): video fps.
            video_path (str): full output path of the video.
            start_frame (Optional[int], optional): the start frame of the stacker. Defaults to None.
            end_frame (Optional[int], optional): the end frame of the stacker.. Defaults to None.
            logger (Optional[Logger], optional): a logging.Logger object for logging use. Defaults to None.
        
        Return:
            int - return code. 0 for success.
        """
        logger = get_useable_logger(logger)
        if start_frame != None or end_frame != None:
            video_loader.reset(start_frame=start_frame, end_frame=end_frame)

        try:
            video_loader.start()
            # 加载示例帧
            sample_frame = video_loader.pop()
            if sample_frame is None:
                return -1
            (h, w, _) = sample_frame.shape
            # 创建容器并打开输出文件
            with av.open(video_path, mode="w") as av_writer:
                # 创建视频流，编码器使用 libx264，帧率为30
                stream = av_writer.add_stream(  # type: ignore
                    "libx264", rate=int(fps))
                stream.width = w
                stream.height = h
                stream.pix_fmt = "yuv420p"  # 通用的视频编码色彩空间

                for _ in range(video_loader.iterations - 1):
                    cur_frame = video_loader.pop()
                    if cur_frame is None: continue
                    # 转换 numpy 数组为 PyAV 的 VideoFrame
                    frame = av.VideoFrame.from_ndarray(
                        cur_frame, format="bgr24")  # type: ignore
                    for packet in stream.encode(frame):
                        av_writer.mux(packet)

                # flush编码器缓存
                for packet in stream.encode():
                    av_writer.mux(packet)

        except Exception as e:
            if logger:
                logger.error(
                    f"Failed to save video {video_path} because: {e.__repr__()}."
                )
            return -1
        finally:
            video_loader.stop()
        return 0
