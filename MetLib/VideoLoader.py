import threading
import time
from .utils import preprocessing


class BaseVideoReader(object):
    def __init__(self, video, iterations, mask, max_poolsize=30) -> None:
        self.video = video
        self.iterations = iterations
        self.mask = mask
        self.max_poolsize = max_poolsize
        self.frame_pool = []


# TODO: 多线程读入似乎会引起意外的异步问题。
# 又：好像是另外一个地方引入的bug。。[捂脸]


class ThreadVideoReader(BaseVideoReader):
    """_summary_

    Args:
        BaseVideoReader (_type_): _description_
    
    To Use ThreadVideoReader By Following Instructions:
    1. init. eg. : T=ThreadVideoReader(args,kwargs)
    2. start eg. : T.start()
    (enter main loop)
    4. pop available frames. e.g.: T.pop(5). 
    (p.s: the class will wait until next reading is allowed (By ThreadLock))
    5. to stop manually or reach the EOF. Whatever, T.stop() is triggered to stop the reading.
    """

    def __init__(self, video, iterations, mask, resize_param,
                 max_poolsize=30) -> None:
        super().__init__(video, iterations, mask, max_poolsize)
        self.resize_param = resize_param
        self.stopped = False
        self.status = True
        self.wait_interval = 0.02
        self.temp_poolsize = 5
        self.temp_pool = []
        self.lock = threading.Lock()

    def start(self):
        self.thread = threading.Thread(target=self.videoloop, args=())
        self.thread.start()
        return self

    def is_popable(self, num):
        if ((len(self.frame_pool) < num) and
            (not self.stopped)) or self.lock.locked():
            return False
        return True

    def pop(self, num):
        while not self.is_popable(num):
            time.sleep(self.wait_interval)
        self.lock.acquire()
        try:
            ret = self.frame_pool[:num]
            self.frame_pool = self.frame_pool[num:]
        finally:
            self.lock.release()
        return ret

    def load_a_frame(self):
        """Load a frame from the video object.

        Returns:
            bool : status code. 1 for success operation, 0 for failure.
        """
        self.status, frame = self.video.read()
        if self.status:
            self.frame = preprocessing(
                frame, mask=self.mask, resize_param=self.resize_param)
            self.temp_pool.append(self.frame)
            return 1
        else:
            self.stop()
            return 0

    def videoloop(self):
        for i in range(
            (self.iterations + self.temp_poolsize - 1) // self.temp_poolsize):
            #print("Enter Main Loop.")
            # wait until frame_pool is not full.
            while len(self.frame_pool) > (
                    self.max_poolsize - self.temp_poolsize) and (
                        not self.stopped):
                #print("Wait for poolsize...")
                time.sleep(self.wait_interval)
            if self.stopped or not self.status: break
            # load several frames
            #print("Load batch of frames...")
            for i in range(self.temp_poolsize):
                if not self.load_a_frame():
                    break
            # append temp_pool to frame_pool with ThreadLock.
            self.lock.acquire()
            try:
                #print("Extending them...")
                self.frame_pool.extend(self.temp_pool)
                self.temp_pool = []
            finally:
                self.lock.release()

    def stop(self):
        self.stopped = True


# TODO: 异步是一种比较优雅的实现 但 不能完全解决文件IO时阻塞的问题。
# 由于编解码依赖ffmpeg/opencv。暂时没有比较好的解决方案。
# 因此，在目前情况下使用异步视频读取时，效率相比阻塞读取没有明显的提升。
# 以及，某些情况下可能会引起意外的未await等错误。
# 目前不建议使用该类别。


class AsyncVideoReader(BaseVideoReader):
    def __init__(self, video, iterations, mask, resize_param, batch=1) -> None:
        super().__init__(video, iterations, mask)
        self.batch = batch
        self.stopped = False
        self.status = False
        self.resize_param = resize_param
        self.cur_iter = 0
        self.temp_pool = []

    def stop(self):
        self.stopped = True

    def preprocessing_pool(self):
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
