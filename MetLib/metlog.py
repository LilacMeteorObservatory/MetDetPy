"""This module is used to manage output stream.
"""
import datetime
import threading
import queue
import sys
import time
from typing import Callable, Optional, cast

try:
    # Reconfigure stdout to utf-8.
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore
except AttributeError:
    sys.stdout.writelines("Unable to set output encoding.")
    sys.stdout.flush()
level_header = [
    "Dropped", "Debug", "Processing", "Info", "Warning", "Error", "Meteor",
    "Fatal"
]

LV_DROPPED = 0
LV_DEBUG = 1
LV_PROCESSING = 2
LV_INFO = 3
LV_WARNING = 4
LV_ERROR = 5
LV_METEOR = 6
LV_FATAL = 7

_STOP_LOGGING = object()


class BaseMetLog(object):

    def __init__(self) -> None:
        pass

    def log(self, level: int, string: str):
        pass

    def debug(self, string: str):
        self.log(LV_DEBUG, string)

    def info(self, string: str):
        self.log(LV_INFO, string)

    def warning(self, string: str):
        self.log(LV_WARNING, string)

    def error(self, string: str):
        self.log(LV_ERROR, string)

    def fatal(self, string: str):
        self.log(LV_FATAL, string)

    def meteor(self, string: str):
        self.log(LV_METEOR, string)

    def dropped(self, string: str):
        self.log(LV_DROPPED, string)

    def processing(self, string: str):
        self.log(LV_PROCESSING, string)

    @property
    def is_empty(self) -> bool:
        return True

    def start(self):
        pass

    def stop(self):
        pass


class ThreadMetLog(BaseMetLog):
    """用于管理输出。
    目前使用多线程的方式进行设计

    Args:
        object (_type_): _description_
        pipe must support `flush`.
    """

    def __init__(self,
                 pipe: Callable[..., None] = print,
                 flush: bool = True,
                 log_level: int = LV_INFO,
                 with_strf: bool = False) -> None:
        # TODO: support other stdout func (like logging)
        self.log_level = log_level
        self.print = pipe
        self.with_strf = with_strf
        self.log_pool: queue.Queue[object] = queue.Queue()
        self.thread = threading.Thread(target=self.log_loop,
                                       name="MetDetPy-Logger",
                                       daemon=False)
        self.stopped = True
        self.flush = flush
        self.wait_interval = 0.02
        self.output_failure_count = 0
        self._fallback_to_stderr = False
        self._state_lock = threading.Lock()

    @property
    def is_empty(self):
        return self.log_pool.empty()

    @property
    def is_stopped(self):
        return self.stopped

    def log_loop(self):
        last_emit_time: Optional[float] = None
        while True:
            cur_log = self.log_pool.get()
            try:
                if cur_log is _STOP_LOGGING:
                    return

                if last_emit_time is not None:
                    next_emit_time = last_emit_time + self.wait_interval
                    remaining = next_emit_time - time.monotonic()
                    if remaining > 0:
                        time.sleep(remaining)

                strf, lv, string = cast(tuple[str, int, str], cur_log)
                self._emit(strf, lv, string)
                last_emit_time = time.monotonic()
            finally:
                self.log_pool.task_done()

    def _write_stderr(self, message: str):
        try:
            sys.stderr.write(f"{message}\n")
            sys.stderr.flush()
        except Exception:
            # There is no safer output channel left. Keep the logging thread
            # alive so a broken pipe cannot terminate the application's log
            # processing or alter its control flow.
            self.output_failure_count += 1

    def _emit(self, strf: str, lv: int, string: str):
        message = f"{strf}{level_header[lv]}: {string}"
        if lv == LV_FATAL or self._fallback_to_stderr:
            self._write_stderr(message)
            return
        try:
            self.print(message, flush=self.flush)
        except Exception as e:
            self.output_failure_count += 1
            self._fallback_to_stderr = True
            self._write_stderr(
                f"Logging output failed ({e!r}); falling back to stderr.")
            self._write_stderr(message)

    def log(self, level: int, string: str):
        if level >= self.log_level:
            time_head = ""
            if self.with_strf:
                time_head = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] - "
            with self._state_lock:
                if not self.stopped:
                    self.log_pool.put((time_head, level, string))

    def start(self):
        with self._state_lock:
            if not self.stopped:
                return
            self.stopped = False
        self.thread.start()

    def stop(self):
        with self._state_lock:
            if self.stopped:
                return
            self.stopped = True
            self.log_pool.put(_STOP_LOGGING)
        self.thread.join()


met_logger = ThreadMetLog()


def set_default_logger(debug_mode: bool, work_mode: str):
    # debug_mode: output LV_DROPPED information.
    # work_mode == backend: LV_DEBUG information only + flush=True force
    # work_mode == frontend: LV_INFO information.
    global met_logger
    if not met_logger.is_stopped:
        met_logger.log(LV_ERROR, "Can not set a running logger.")
        return -1
    if debug_mode:
        met_logger.log_level = LV_DROPPED
        met_logger.with_strf = True
    elif work_mode == "backend":
        met_logger.flush = True
        met_logger.log_level = LV_DROPPED
        level_header[LV_DROPPED] = "Meteor"
    else:
        met_logger.log_level = LV_INFO
    return 0


def get_default_logger():
    return met_logger


def get_useable_logger(logger: Optional[BaseMetLog]) -> BaseMetLog:
    if logger is None:
        return BaseMetLog()
    return logger
