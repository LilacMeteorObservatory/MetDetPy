import os
import subprocess
import sys
import tempfile
from abc import ABCMeta
from typing import Any, Optional, Sequence, Union

import av
import cv2

from .metlog import BaseMetLog, get_useable_logger
from .metstruct import ExportOption, FFMpegConfig
from .utils import PLATFORM_MAPPING, U8Mat, frame2ts, transpose_wh, WORK_PATH
from .videoloader import VanillaVideoLoader

platform = PLATFORM_MAPPING[sys.platform]
exec_suffix = ""
if (platform == "win"):
    exec_suffix = ".exe"

# the first in the list represents the default
CONTAINER_AUDIO_ACCEPT = {
    'mp4': ['aac', 'mp3', 'ac3'],
    'm4a': ['aac', 'mp3', 'ac3'],
    'mov': ['aac', 'mp3', 'ac3', 'pcm_s16le'],
    'mkv': ['aac', 'mp3', 'ac3', 'vorbis', 'opus', 'flac', 'pcm_s16le'],
    'webm': ['vorbis', 'opus'],
    'avi': ['mp3', 'pcm_s16le', 'ac3'],
    'wav': ['pcm_s16le', 'pcm_s24le', 'flac'],
}


def _chk_ffmpeg_status(exec: str) -> bool:
    """
    Use -version to probe availability; accept returncode 0
    """
    try:
        p1 = subprocess.run([exec, "-version"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
        return p1.returncode == 0
    except (FileNotFoundError, Exception):
        return False


class SeriesLoader(object):
    """A wrapper that mocks a video series to a VideoLoader-like class.
    """

    def __init__(self,
                 video_series: Sequence[U8Mat],
                 fps: float,
                 video_name: Optional[str] = None):
        # 如果需要导出片段对应原始音频，则需要指定视频名称和起始帧
        self.video_name = video_name
        self.video_series = video_series
        self.fps = fps
        self.cur_index = -1

    def pop(self):
        if self.cur_index >= self.iterations - 1:
            return None
        self.cur_index += 1
        return self.video_series[self.cur_index]

    def start(self):
        pass

    def stop(self):
        pass

    def reset(self,
              start_frame: Optional[int] = None,
              end_frame: Optional[int] = None):
        pass

    @property
    def iterations(self):
        return len(self.video_series)

    @property
    def runtime_size(self):
        return transpose_wh(self.video_series[0].shape[:2])


class BaseVideoWriter(ABCMeta):
    """
    VideoWriter a class that provides class methods about video writing.
    
    A video writer must support:
    1. save_video: save a video from a given Sequence.
    2. save_video_by_stream: execute a stream copy from the source video to the target.
    3. save_video_with_audio: save a video from a given Sequence, while copying the audio from the source video.
    """

    @classmethod
    def save_video(cls,
                   video_series: Sequence[U8Mat],
                   fps: Union[int, float],
                   export_option: ExportOption,
                   video_path: str,
                   logger: Optional[BaseMetLog] = None,
                   *args: Any,
                   **kwargs: Any) -> int:
        series_loader = SeriesLoader(video_series, fps)
        return cls.save_video_by_stream(series_loader,
                                        export_option,
                                        video_path,
                                        logger=logger)

    @classmethod
    def save_video_by_stream(cls,
                             video_loader: Union[VanillaVideoLoader,
                                                 SeriesLoader],
                             export_option: ExportOption,
                             video_path: str,
                             start_frame: Optional[int] = None,
                             end_frame: Optional[int] = None,
                             logger: Optional[BaseMetLog] = None) -> int:
        raise NotImplementedError("...")

    @classmethod
    def save_video_with_audio(cls,
                              video_series: Sequence[U8Mat],
                              video_loader: Union[VanillaVideoLoader,
                                                  SeriesLoader],
                              export_option: ExportOption,
                              video_path: str,
                              start_frame: Optional[int] = None,
                              end_frame: Optional[int] = None,
                              logger: Optional[BaseMetLog] = None) -> int:
        """By default, videowriter does not support save video with source audio.
        Thus you don't have to implement this method except your writer intends to do so.
        """
        logger = get_useable_logger(logger)
        logger.warning(
            f"{cls.__name__} does not support save video with source audio. "
            "The output video is without audio.")
        return cls.save_video(video_series, video_loader.fps, export_option,
                              video_path, logger)


class OpenCVVideoWriter(BaseVideoWriter):

    @classmethod
    def save_video_by_stream(cls,
                             video_loader: Union[VanillaVideoLoader,
                                                 SeriesLoader],
                             export_option: ExportOption,
                             video_path: str,
                             start_frame: Optional[int] = None,
                             end_frame: Optional[int] = None,
                             logger: Optional[BaseMetLog] = None) -> int:
        """ Save video by stream (to avoid OutOfMemory) via OpenCV.
        
        NOTE:
        1. OpenCVVideoWriter only supports writing .AVI video file. Thus no export option will work for this.
        2. OpenCVVideoWriter is not able to copy audio stream.

        Args:
            video_loader (VanillaVideoLoader): initialized video loader.
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
        if not video_path.lower().endswith("avi"):
            logger.fatal(
                f"Failed to save video, because {cls.__name__} only support writing .avi video file."
            )
            return -1
        cv_writer = None
        try:
            video_loader.start()
            cv_writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*"MJPG"),  # type: ignore
                video_loader.fps,
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
    def save_tmp_avi(cls, video_series: Sequence[U8Mat], fps: Union[int,
                                                                    float],
                     export_option: ExportOption, logger: BaseMetLog):
        tmpf = None
        try:
            fd, tmpf = tempfile.mkstemp(suffix='.avi')
            os.close(fd)
            logger.info(f"Writing temporary AVI to '{tmpf}' via PyAV.")
            rc = cls.save_video(video_series, fps, export_option, tmpf, logger)
            if rc != 0:
                logger.error(f"{cls.__name__} failed to write temporary AVI.")
                raise Exception("return code != 0")
            return tmpf
        except Exception as e:
            if tmpf and os.path.exists(tmpf):
                try:
                    os.remove(tmpf)
                    logger.debug(f"Removed temporary file {tmpf}.")
                except Exception:
                    logger.warning(f"Failed to remove temporary file {tmpf}.")
            raise e

    @classmethod
    def save_video_by_stream(cls,
                             video_loader: Union[VanillaVideoLoader,
                                                 SeriesLoader],
                             export_option: ExportOption,
                             video_path: str,
                             start_frame: Optional[int] = None,
                             end_frame: Optional[int] = None,
                             logger: Optional[BaseMetLog] = None) -> int:
        """ Save video by stream (to avoid OutOfMemory).
        
        NOTE:
        1. PyAVVideoWriter encoding support varies, according to its binding FFMpeg option.
            (for release version, its binding FFMpeg does not support x264/265 encodc, due to 
            the license restriction. If you want to encoding h264 video with PyAVVideoWriter, 
            you should install PyAV with your own FFMpeg binded.)
        2. Since it does not support x264/265, audio copying in PyAVVideoWriter is ignored for now.

        Args:
            video_loader (VanillaVideoLoader): initialized video loader.
            fps (Union[int, float]): video fps.
            video_path (str): full output path of the video.
            start_frame (Optional[int], optional): the start frame of the stacker. Defaults to None.
            end_frame (Optional[int], optional): the end frame of the stacker. Defaults to None.
            logger (Optional[Logger], optional): a logging.Logger object for logging use. Defaults to None.
        
        Return:
            int - return code. 0 for success.
        """
        logger = get_useable_logger(logger)
        if start_frame != None or end_frame != None:
            video_loader.reset(start_frame=start_frame, end_frame=end_frame)

        try:
            video_loader.start()
            w, h = video_loader.runtime_size
            with av.open(video_path, mode="w") as av_writer:
                stream: av.VideoStream = av_writer.add_stream(  # type: ignore
                    export_option.ffmpeg_config.video_encoder,
                    rate=int(video_loader.fps))
                stream.width = w
                stream.height = h
                stream.pix_fmt = export_option.ffmpeg_config.pix_fmt

                for _ in range(video_loader.iterations):
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


class FFMpegVideoWriter(BaseVideoWriter):

    @classmethod
    def _chk_ffmpeg_path(cls, ffmpeg_config: FFMpegConfig) -> FFMpegConfig:
        """
        Check availability of ffmpeg and ffprobe, in the following order:
        1. if `ffmpeg_path` is provided, try executables under that path;
        2. try the root path of `MetDetPy` project;
        3. rely on PATH.

        fill ffmpeg_path and ffprobe_path with executable one. If none of paths works,
        raise FileNotFoundError.
        """
        FFMPEG_NAME = f"ffmpeg{exec_suffix}"
        FFPROBE_NAME = f"ffprobe{exec_suffix}"
        input_path = ffmpeg_config.path
        if input_path:
            if (_chk_ffmpeg_status(os.path.join(input_path, FFMPEG_NAME))
                    and _chk_ffmpeg_status(
                        os.path.join(input_path, FFPROBE_NAME))):
                ffmpeg_config.ffmpeg_path = os.path.join(
                    input_path, FFMPEG_NAME)
                ffmpeg_config.ffprobe_path = os.path.join(
                    input_path, FFPROBE_NAME)
                return ffmpeg_config
        if (_chk_ffmpeg_status(os.path.join(WORK_PATH, FFMPEG_NAME))
                and _chk_ffmpeg_status(os.path.join(WORK_PATH, FFPROBE_NAME))):
            ffmpeg_config.ffmpeg_path = os.path.join(WORK_PATH, FFMPEG_NAME)
            ffmpeg_config.ffprobe_path = os.path.join(WORK_PATH, FFPROBE_NAME)
            return ffmpeg_config
        if (_chk_ffmpeg_status(FFMPEG_NAME)
                and _chk_ffmpeg_status(FFPROBE_NAME)):
            ffmpeg_config.ffmpeg_path = FFMPEG_NAME
            ffmpeg_config.ffprobe_path = FFPROBE_NAME
            return ffmpeg_config
        raise FileNotFoundError("FFMpeg or FFProbe is unavilable.")

    @classmethod
    def _get_audio_args(cls, ffprobe_exe: str, src: str, tgt: str):
        """
        Probe first audio stream codec name from src using ffprobe.
        Returns codec_name string (e.g. 'pcm_s16le', 'aac', 'mp3', 'opus', ...) or None if no audio.
        """
        codec = None
        try:
            proc = subprocess.run([
                ffprobe_exe, '-v', 'error', '-select_streams', 'a:0',
                '-show_entries', 'stream=codec_name', '-of',
                'default=noprint_wrappers=1:nokey=1', src
            ],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  text=True,
                                  check=False)
            codec_str = proc.stdout.strip()
            if codec_str != "":
                codec = codec_str.splitlines()[0].strip()
        except Exception:
            pass

        audio_args = ['-c:a', 'copy']
        if codec is not None:
            output_accept_list = CONTAINER_AUDIO_ACCEPT[tgt.lower().split(".")
                                                        [-1]]
            if not codec in output_accept_list:
                audio_args = ['-c:a', output_accept_list[0], '-b:a', '192k']
        return audio_args

    @classmethod
    def _estimate_k_frame(cls,
                          ffprobe_exe: str,
                          video_path: str,
                          start_time: float,
                          logger: BaseMetLog,
                          max_retry_cnt: int = 5):
        """Use ffprobe to find the nearest previous I-frame time K"""
        cur_cnt = 0
        rev_time = 2
        eps = 1e-6
        cur_start_time = start_time
        while cur_cnt < max_retry_cnt:
            cur_cnt += 1
            logger.debug(f"Attempt {cur_cnt}/{max_retry_cnt}")
            try:
                cmd_probe = [
                    ffprobe_exe, '-v', 'error', '-select_streams', 'v:0',
                    '-show_entries', 'frame=key_frame,pkt_pts_time',
                    '-read_intervals',
                    f'{max(0, cur_start_time-rev_time):.3f}%{cur_start_time:.3f}',
                    '-of', 'csv=p=0', video_path
                ]
                logger.debug(f"Call ffprobe: {' '.join(cmd_probe)}")
                res = subprocess.run(cmd_probe,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     text=True)
                lines = res.stdout.splitlines()
                parts = [[
                    p.strip() for p in line.split(",") if p.strip() != ""
                ] for line in lines if line != ""]
                collect_ks = [
                    float(p[1]) for p in parts
                    if p[0] == "1" and float(p[1]) < start_time + eps
                ]
                if len(collect_ks) > 0:
                    return max(collect_ks)
            except Exception as e:
                continue
            logger.warning(
                f"attempt failed to find keyframes in this iteration.")
            cur_start_time = cur_start_time - rev_time + eps
            rev_time *= 2
        # all attempts failed.
        logger.warning(
            f"ffprobe failed to list keyframes. Falling back to heuristic K=S-2s."
        )
        return max(0.0, start_time - 2.0)

    @classmethod
    def save_video(cls,
                   video_series: Sequence[U8Mat],
                   fps: Union[int, float],
                   export_option: ExportOption,
                   video_path: str,
                   logger: Optional[BaseMetLog] = None,
                   *args: Any,
                   **kwargs: Any) -> int:
        logger = get_useable_logger(logger)
        # PyAV directly handling AVI format.
        if video_path.lower().endswith('.avi'):
            return PyAVVideoWriter.save_video(video_series, fps, export_option,
                                              video_path, logger)

        config = cls._chk_ffmpeg_path(export_option.ffmpeg_config)
        assert config.ffmpeg_path and config.ffprobe_path
        tmpf = None
        try:
            # write a temporary .avi via PyAV, then transcode with ffmpeg
            tmpf = PyAVVideoWriter.save_tmp_avi(video_series, fps,
                                                export_option, logger)
            ff_cmd: list[str] = [
                config.ffmpeg_path, '-i', tmpf, '-c:v', config.video_encoder,
                '-preset', config.preset, '-crf',
                str(config.crf), '-pix_fmt', config.pix_fmt, '-an', '-y',
                video_path
            ]
            logger.info(
                f"Running ffmpeg for transcoding temporary AVI: {' '.join(ff_cmd)}"
            )
            p = subprocess.run(ff_cmd,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               text=True)
            if p.returncode != 0:
                logger.error(
                    f"ffmpeg transcode failed (rc={p.returncode}). stderr: {p.stderr}"
                )
                return -1
        finally:
            if tmpf and os.path.exists(tmpf):
                try:
                    os.remove(tmpf)
                    logger.debug(f"Removed temporary file {tmpf}.")
                except Exception:
                    logger.warning(f"Failed to remove temporary file {tmpf}.")
        return 0

    @classmethod
    def save_video_by_stream(cls,
                             video_loader: Union[VanillaVideoLoader,
                                                 SeriesLoader],
                             export_option: ExportOption,
                             video_path: str,
                             start_frame: Optional[int] = None,
                             end_frame: Optional[int] = None,
                             logger: Optional[BaseMetLog] = None) -> int:
        logger = get_useable_logger(logger)
        # in this implementation, video loader is not actually called, only metadata is used.
        if start_frame != None or end_frame != None:
            video_loader.reset(start_frame=start_frame, end_frame=end_frame)

        # check ffmpeg availability, cfg loading
        config = cls._chk_ffmpeg_path(export_option.ffmpeg_config)
        logger.info(
            f"Found ffmpeg at '{config.ffmpeg_path}'; ffprobe at '{config.ffprobe_path}'"
        )

        # check video availability
        assert isinstance(video_loader, VanillaVideoLoader)
        assert config.ffmpeg_path and config.ffprobe_path
        input_path = video_loader.video_name
        if not input_path or not os.path.exists(input_path):
            logger.fatal(
                f"Failed to save video: input file not found: {input_path}")
            return -1
        S, E = video_loader.start_time / 1000, video_loader.end_time / 1000
        # Use ffprobe to find the nearest previous I-frame time K
        logger.debug(f"Start K-frame seeking...")
        K = cls._estimate_k_frame(config.ffprobe_path, input_path, S, logger)
        logger.debug(f"K-frame seeking finished. got timestamp at {K:.3f}")

        # get input audio encodec
        audio_args = cls._get_audio_args(config.ffprobe_path,
                                             src=video_loader.video_name,
                                             tgt=video_path)

        # Build ffmpeg command: coarse seek (-ss K) + precise seek (-ss offset) + transcode video + copy audio
        # Video transcode parameters (recommended / tunable):
        #   -c:v <encoder>   (from export_option.video_encoder, default libx264)
        #   -preset <preset> (tradeoff speed vs compression: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower)
        #   -crf <value>     (quality for x264: lower is better quality; common range 18-28)
        #   -pix_fmt <fmt>   (pixel format, e.g. yuv420p)
        # Audio is copied with -c:a copy to preserve original audio stream.
        ff_cmd: list[str] = [
            config.ffmpeg_path, '-ss', f"{K:.3f}", '-i', input_path, '-ss',
            f"{(S - K):.3f}", '-t', f"{(E - S):.3f}", '-c:v',
            config.video_encoder, '-preset', config.preset, '-crf',
            str(config.crf), '-pix_fmt', config.pix_fmt, *audio_args,
            '-avoid_negative_ts', '1', '-y', video_path
        ]

        try:
            logger.info(f"Running ffmpeg command: {' '.join(ff_cmd)}")
            p = subprocess.run(ff_cmd,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               text=True)
            if p.returncode != 0:
                logger.error(
                    f"ffmpeg failed (rc={p.returncode}). stderr: {p.stderr}")
                return -1
        except Exception as e:
            logger.error(f"ffmpeg execution failed: {e}")
            return -1

        return 0

    @classmethod
    def save_video_with_audio(cls,
                              video_series: Sequence[U8Mat],
                              video_loader: Union[VanillaVideoLoader,
                                                  SeriesLoader],
                              export_option: ExportOption,
                              video_path: str,
                              start_frame: Optional[int] = None,
                              end_frame: Optional[int] = None,
                              logger: Optional[BaseMetLog] = None) -> int:
        logger = get_useable_logger(logger)
        config = cls._chk_ffmpeg_path(export_option.ffmpeg_config)
        assert config.ffmpeg_path and config.ffprobe_path
        assert isinstance(video_loader, VanillaVideoLoader)
        tmpf = None
        try:
            # write a temporary .avi via PyAV, then transcode with ffmpeg
            tmpf = PyAVVideoWriter.save_tmp_avi(video_series, video_loader.fps,
                                                export_option, logger)
            # 从原始文件读取音频，把原始音频作为第二输入并映射
            src = video_loader.video_name
            if not src:
                logger.error("Lack video name.")
                return -1
            start_frame = start_frame if start_frame is not None else video_loader.start_frame
            end_frame = end_frame if end_frame is not None else video_loader.end_frame
            # duration in seconds
            duration = (end_frame - start_frame) / float(video_loader.fps)
            # start timestamp string for ffmpeg (frame2ts returns hh:mm:ss.xxx)
            start_ts = frame2ts(start_frame, video_loader.fps)

            # get input audio encodec
            audio_args = cls._get_audio_args(config.ffprobe_path,
                                                src=video_loader.video_name,
                                                tgt=video_path)
            
            # Map video from tmp (input 0) and audio from source (input 1)
            # Use optional audio map '1:a:0?' to tolerate missing audio track
            # Seek and trim the source audio as input options for input 1
            # Place -ss and -t before the second '-i' so they apply to that input
            ff_cmd = [
                config.ffmpeg_path, '-i', tmpf, '-ss', f"{start_ts}", '-t',
                f"{duration:.3f}", '-i', src, '-map', '0:v:0', '-map',
                '1:a:0?', '-c:v', config.video_encoder, '-preset',
                config.preset, '-crf',
                str(config.crf), '-pix_fmt', config.pix_fmt, *audio_args,
                '-avoid_negative_ts', '1', '-y', video_path
            ]

            try:
                logger.info(f"Running ffmpeg command: {' '.join(ff_cmd)}")
                p = subprocess.run(ff_cmd,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   text=True)
                if p.returncode != 0:
                    logger.error(
                        f"ffmpeg failed (rc={p.returncode}). stderr: {p.stderr}"
                    )
                    return -1
            except Exception as e:
                logger.error(f"ffmpeg execution failed: {e}")
                return -1
            return 0
        finally:
            if tmpf and os.path.exists(tmpf):
                try:
                    os.remove(tmpf)
                    logger.debug(f"Removed temporary file {tmpf}.")
                except Exception:
                    logger.warning(f"Failed to remove temporary file {tmpf}.")
