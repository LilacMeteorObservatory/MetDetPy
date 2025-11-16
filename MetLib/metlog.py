"""This module is used to manage output stream.
"""
import time
import datetime
import threading
import queue
import sys
from typing import Callable, Optional

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
        self.log_pool: queue.Queue[tuple[str, int, str]] = queue.Queue()
        self.thread = threading.Thread(target=self.log_loop, args=())
        self.stopped = True
        self.flush = flush
        self.wait_interval = 0.02

    @property
    def is_empty(self):
        return self.log_pool.empty()

    @property
    def is_stopped(self):
        return self.stopped

    def log_loop(self):
        while not (self.stopped and self.is_empty):
            time.sleep(self.wait_interval)
            cur_log = self.log_pool.get()
            strf, lv, string = cur_log
            if lv == LV_FATAL:
                sys.stderr.write(f"{strf}{level_header[lv]}: {string}\n")
                sys.stderr.flush()
            else:
                self.print(f"{strf}{level_header[lv]}: {string}", flush=self.flush)
        self.log_pool.task_done()

    def log(self, level: int, string: str):
        if level >= self.log_level:
            time_head = ""
            if self.with_strf:
                time_head = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] - "
            self.log_pool.put((time_head, level, string))

    def start(self):
        self.stopped = False
        self.thread.setDaemon(True)
        self.thread.start()

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=2)


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
