import threading
import asyncio
import time
from .utils import preprocessing


class BaseVideoReader(object):
    def __init__(self, video, iterations, mask) -> None:
        self.video = video
        self.iterations = iterations
        self.mask = mask
        self.frame_pool = []


# TODO: 目前的读入硬编码参数比较多。酌情在后期参数化他们。
# TODO: 多线程读入似乎会引起意外的异步问题。
# 又：好像是另外一个地方引入的bug。。[捂脸]


class ThreadVideoReader(BaseVideoReader):
    def __init__(self, video, iterations, mask, resize_param) -> None:
        super().__init__(video, iterations, mask)
        self.resize_param = resize_param
        self.stopped = False
        self.status = False
        self.load_a_frame()

    def start(self):
        self.thread = threading.Thread(target=self.get, args=())
        self.thread.start()
        return self

    def load_a_frame(self):
        temp_pool = []
        for t in range(10):
            self.status, frame = self.video.read()
            if self.status:
                self.frame = preprocessing(
                    frame, mask=self.mask, resize_param=self.resize_param)
                temp_pool.append(self.frame)
            else:
                self.stop()
                break
        self.frame_pool.extend(temp_pool)

    def get(self):
        for i in range(self.iterations):
            while len(self.frame_pool) > 30:
                time.sleep(0.1)
            if self.stopped or not self.status: break
            self.load_a_frame()

    def stop(self):
        self.stopped = True


class AsyncVideoReader(BaseVideoReader):
    def __init__(self, video, iterations, mask, resize_param, batch=1) -> None:
        super().__init__(video, iterations, mask)
        self.batch = batch
        self.stopped = False
        self.status = False
        self.resize_param = resize_param
        self.cur_iter = 0
        self.max_poolsize = 30
        self.temp_pool = []

    def stop(self):
        self.stopped = True

    def preprocessing_pool(self):
        #self.frame_pool.extend([
        #    preprocessing(
        #        frame, mask=self.mask, resize_param=self.resize_param)
        #    for frame in self.temp_pool
        #])
        self.frame_pool.extend(self.temp_pool)
        self.temp_pool = []

    async def read_a_batch(self):
        #self.cur_iter += 1
        if (self.cur_iter > self.iterations) or (len(self.frame_pool) >
                                                 self.max_poolsize):
            return
        for n in range(self.batch):
            self.status, frame = self.video.read()
            if not self.status:
                self.stop()
                break
            self.temp_pool.append(
                preprocessing(
                    frame, mask=self.mask, resize_param=self.resize_param))
