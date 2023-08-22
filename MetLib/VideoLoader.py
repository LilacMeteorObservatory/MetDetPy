import threading
import time

from .utils import img_max, m3func, mix_max_median_stacker

available_func = dict(max=img_max, m3func=m3func, mix=mix_max_median_stacker)


class BaseVideoReader(object):
    """ 
    # BaseVideoReader
    This class is used to load the video from the file.

    In this basic implementation, video are loaded every time .pop() method is called, 
    which is an block file IO implementation.

    ## Args:

        video (Any): The video object that supports .read() method to load the next frame. 
                    We recommend to use cv2.VideoCapture object.
        iterations (int): The number of frames that are going to load.
        pre_func (func): the preprocessing function that only takes frames[ndarray] 
                    as the only arguments. You can use functools.partical to 
                    construct such a function.
        max_poolsize (int, optional): the max size of the frame buffer. Defaults to 30.
    
    ## Usage

    All of VideoReader (take T=VideoReader(args,kwargs) as an example) classes should be designed and 
    utilized following these instructions:
    
    1. Call .start() method before using it. eg. : T.start()
    2. Pop the number of frames from its frame_pool with the .pop() method. e.g.: T.pop(num=5). 
    3. when its video reaches the EOF or an exception is raised, its .stop() method should be triggered. 
       Then T.stopped will be set to True to ensure other parts of the program be terminated normally.
    """

    def __init__(self,
                 video,
                 start_frame,
                 iterations,
                 pre_func,
                 exp_frame,
                 merge_func,
                 max_poolsize=30):
        """    This class is used to load the video from the file.
        Args:
            video (Any): The video object that supports .read() method to load the next frame. 
                         We recommend to use cv2.VideoCapture object.
            iterations (int): The number of frames that are going to load.
            pre_func (func): the preprocessing function that only takes frames[ndarray] as the only arguments. 
                             You can use functools.partical to construct such a function.
            max_poolsize (int, optional): the max size of the frame buffer. Defaults to 30.

        """
        self.video = video
        self.pre_func = pre_func
        self.iterations = iterations
        self.start_frame = start_frame
        self.max_poolsize = max_poolsize
        self.status = True
        self.stopped = False
        self.exp_frame = exp_frame
        self.frame_pool = []

        if not merge_func in available_func:
            raise NameError(
                f"Unsupported preprocessing merge function name: {merge_func}."
            )
        self.merge_func = available_func[merge_func]

    def start(self):
        self.cur_iter = self.iterations
        self.video.set_to(self.start_frame)

    def reset(self, start_frame=None, iterations=None, exp_frame=None):
        """重置并允许VideoLoader读取另一个片段。
        Args:
            frame (_type_): _description_
        """
        if iterations != None:
            self.iterations = iterations
        if start_frame != None:
            self.start_frame = start_frame
        if exp_frame != None:
            self.exp_frame = exp_frame
        self.stopped = False

    def pop(self):
        self.frame_pool = []
        for _ in range(self.exp_frame):
            status, frame = self.video.read()
            if status:
                self.frame_pool.append(self.pre_func(frame))
            else:
                self.stop()
                break
        self.cur_iter -= self.exp_frame
        if self.cur_iter <= 0: self.stop()

        if self.exp_frame == 1:
            return self.frame_pool[0]
        return self.merge_func(self.frame_pool)

    def stop(self):
        self.stopped = True


class ThreadVideoReader(BaseVideoReader):
    """ 
    # ThreadVideoReader
    This class is used to load the video from the file with an independent thread.  
    On average, ThreadVideoReader provides about 50% speedup.

    ## Args:

        video (Any): The video object that supports .read() method to load the next frame. 
                    We recommend to use cv2.VideoCapture object.
        iterations (int): The number of frames that are going to load.
        pre_func (func): the preprocessing function that only takes frames[ndarray] 
                    as the only arguments. You can use functools.partical to 
                    construct such a function.
        max_poolsize (int, optional): the max size of the frame buffer. Defaults to 30.
    
    ## Usage

    All of VideoReader (take T=VideoReader(args,kwargs) as an example) classes should be designed and 
    utilized following these instructions:
    
    1. Call .start() method before using it. eg. : T.start()
    2. Pop the number of frames from its frame_pool with the .pop() method. e.g.: T.pop(num=5). 
    3. when its video reaches the EOF or an exception is raised, its .stop() method should be triggered. 
       Then T.stopped will be set to True to ensure other parts of the program be terminated normally.
    """

    def __init__(self,
                 video,
                 start_frame,
                 iterations,
                 pre_func,
                 exp_frame,
                 merge_func,
                 max_poolsize=30) -> None:
        super().__init__(video, start_frame, iterations, pre_func, exp_frame,
                         merge_func, max_poolsize)
        self.wait_interval = 0.02
        self.lock = threading.Lock()

    @property
    def is_empty(self) -> bool:
        return len(self.frame_pool) == 0

    def start(self):
        self.frame_pool = []
        self.stopped = False
        self.status = True
        self.video.set_to(self.start_frame)
        self.thread = threading.Thread(target=self.videoloop, args=())
        self.thread.setDaemon(True)
        self.thread.start()
        return self

    def reset(self, start_frame=None, iterations=None, exp_frame=None):
        if isinstance(start_frame, int) and start_frame >= 0:
            self.start_frame = start_frame
            self.video.set_to(start_frame)
        super().reset(iterations=iterations, exp_frame=exp_frame)

    def is_popable(self, num):
        """是否能够提供num个帧。
        1. 如果被锁（某线程试图更新池）或池中完全没有新帧，则False。
        2. 剩余池中帧数有但小于num，同时已经读取完毕，则True。
        3. 其他情况一定能提供需要的帧数。
        """
        if (len(self.frame_pool) == 0 and self.stopped):
            # TODO: 当video替换为原始video名称时，此处需要做对应改动（所有正常结束区域应当释放）
            self.video.release()
            self.thread.join()
            raise TimeoutError(
                "ReadError: Attempt to read frame(s) from an ended VideoReader object."
            )
        if len(self.frame_pool) == 0:
            return False
        if (len(self.frame_pool) < num and (not self.stopped)):
            return False
        return True

    def pop(self):
        while (not self.is_popable(self.exp_frame)):
            time.sleep(self.wait_interval)
        self.lock.acquire()
        ret = self.frame_pool[:self.exp_frame]
        self.frame_pool = self.frame_pool[self.exp_frame:]
        self.lock.release()

        if self.exp_frame == 1:
            return ret[0]
        return self.merge_func(ret)

    def load_a_frame(self):
        """Load a frame from the video object.

        Returns:
            bool : status code. 1 for success operation, 0 for failure.
        """
        self.status, frame = self.video.read()
        if self.status:
            self.lock.acquire()
            self.frame_pool.append(self.pre_func(frame))
            self.lock.release()
            return True
        else:
            self.stop()
            return False

    def videoloop(self):
        try:
            for i in range(self.iterations):
                # wait until frame_pool is not full.
                while len(self.frame_pool) > self.max_poolsize and (
                        not self.stopped):
                    time.sleep(self.wait_interval)
                if self.stopped or not self.status: break
                if not self.load_a_frame():
                    break
        finally:
            self.stop()


# TODO: 异步是一种更优雅的实现 但由于目前的实现在编解码步骤依赖ffmpeg/opencv，不能完全解决文件IO时阻塞的问题。
# 目前情况下使用异步视频读取时，效率相比阻塞读取没有明显的提升。该部分代码目前暂时弃置不更新。


class AsyncVideoReader(BaseVideoReader):

    def __init__(self, video, iterations, pre_func, batch=1) -> None:
        super().__init__(video, iterations, pre_func)
        self.batch = batch
        self.cur_iter = 0
        self.last_batch = None

    def preprocessing_pool(self):
        self.frame_pool.extend(self.temp_pool)

    def start(self):
        self.last_batch = self.read_a_batch()

    async def pop(self, num):
        await self.last_batch
        while (len(self.frame_pool) < num) and (not self.stopped):
            await self.read_a_batch()
        ret = self.frame_pool[:num]
        self.frame_pool = self.frame_pool[num:]
        if not self.stopped:
            self.last_batch = self.read_a_batch()
        # TODO: this should not be return.....
        return ret

    async def read_a_batch(self):
        if (self.cur_iter > self.iterations) or (len(self.frame_pool) >
                                                 self.max_poolsize):
            return
        self.cur_iter += self.batch
        for n in range(self.batch):
            self.status, frame = self.video.read()
            if not self.status:
                self.stop()
                break
            self.frame_pool.append(self.pre_func(frame))
