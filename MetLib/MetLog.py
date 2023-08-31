"""This module is used to manage output stream.
"""
import time
import threading
import queue

level_header = ("Dropped", "Debug","Processing","Info", "Warning", "Error", "Meteor")

LV_DROPPED = 0
LV_DEBUG = 1
LV_PROCESSING=2
LV_INFO = 3
LV_WARNING = 4
LV_ERROR = 5
LV_METEOR = 6



class BaseMetLog(object):

    def __init__(self) -> None:
        pass
    
    def log(self, level, string):
        pass

    def debug(self, string):
        self.log(LV_DEBUG, string)
    
    def info(self, string):
        self.log(LV_INFO, string)
    
    def warning(self, string):
        self.log(LV_WARNING, string)
    
    def error(self, string):
        self.log(LV_ERROR, string)

    def meteor(self, string):
        self.log(LV_METEOR, string)

    def dropped(self, string):
        self.log(LV_DROPPED, string)
    
    def processing(self, string):
        self.log(LV_PROCESSING, string)


class ThreadMetLog(BaseMetLog):
    """用于管理输出。
    目前使用多线程的方式进行设计

    Args:
        object (_type_): _description_
        pipe must support `flush`.
    """

    def __init__(self, pipe=print,flush=True, log_level=LV_INFO) -> None:
        self.log_level = log_level
        self.print = pipe
        self.log_pool = queue.Queue()
        self.thread = threading.Thread(target=self.log_loop, args=())
        self.stopped = True
        # how to support other stdout func?
        # TODO: fix this.
        self.flush=flush
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
            lv, string = cur_log
            self.print(f"{level_header[lv]}: {string}",flush=self.flush)
        self.log_pool.task_done()

    def log(self, level, string):
        if level >= self.log_level:
            self.log_pool.put([level, string])

    def start(self):
        self.stopped = False
        self.thread.setDaemon(True)
        self.thread.start()

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=2)

met_logger = ThreadMetLog()


def set_default_logger(debug_mode, work_mode):
    # debug_mode: output LV_DROPPED information.
    # work_mode == backend: LV_DEBUG information only + flush=True force
    # work_mode == frontend: LV_INFO information.
    global met_logger
    if not met_logger.is_stopped:
        met_logger.log(LV_ERROR, "Can not set a running logger.")
        return -1
    if debug_mode:
        met_logger.log_level=LV_DROPPED
    elif work_mode=="backend":
        met_logger.flush=True
        met_logger.log_level=LV_DEBUG
    else:
        met_logger.log_level=LV_INFO

    return 0


def get_default_logger():
    return met_logger