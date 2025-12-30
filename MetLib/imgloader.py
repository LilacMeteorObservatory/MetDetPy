"""
ImgLoader handles the whole image loading process, including:

1. reading different image formats (raw, PNG, JPEG) from disk
2. mask loading and applying
3. preprocessing pipeline (e.g., contrast stretching, scaling)

"""

import multiprocessing as mp
import queue
import threading
from abc import ABCMeta, abstractmethod
from typing import Any, Optional, Union

from .fileio import (SUPPORT_COMMON_FORMAT, SUPPORT_RAW_FORMAT, is_ext_within,
                     load_8bit_image, load_raw_with_preprocess)
from .metlog import BaseMetLog, get_default_logger
from .utils import U8Mat

ImgPair = tuple[Union[str, None], Union[U8Mat, None]]

# processing RAW could be really long...
IMG_GET_TIMEOUT = 10
MT_HEART_TIME = 1
DEFAULT_WORKER_NUM = mp.cpu_count() // 2


class BaseImgLoader(metaclass=ABCMeta):

    def __init__(self, img_fn_list: list[str]) -> None:
        self.img_fn_list = img_fn_list

    @property
    def num_images(self) -> int:
        return len(self.img_fn_list)

    def start(self):
        pass

    def stop(self):
        pass

    @abstractmethod
    def pop(self) -> ImgPair:
        pass

    def summary(self) -> str:
        return f"{self.__class__.__name__} with {self.num_images} images."


class VanillaImgLoader(BaseImgLoader):

    def __init__(self,
                 img_fn_list: list[str],
                 raw_power: float = 2.222,
                 target_nl_mean: float = 0.3,
                 contrast_alpha: float = 1.2,
                 output_bps: int = 8,
                 logger: BaseMetLog = get_default_logger(),
                 **kwargs: Any) -> None:
        super().__init__(img_fn_list)
        self.output_bps = output_bps
        self.raw_power = raw_power
        self.target_nl_mean = target_nl_mean
        self.contrast_alpha = contrast_alpha
        self.logger = logger
        self.current_idx = -1

    def pop(self):
        return self._pop()

    def _pop(self):
        img_fname, img = None, None
        self.current_idx += 1
        if self.current_idx >= self.num_images:
            return (None, None)
        img_fname = self.img_fn_list[self.current_idx]
        try:
            if is_ext_within(img_fname, SUPPORT_RAW_FORMAT):
                img = load_raw_with_preprocess(
                    img_fname,
                    power=self.raw_power,
                    target_nl_mean=self.target_nl_mean,
                    contrast_alpha=self.contrast_alpha,
                    output_bps=8 if self.output_bps == 8 else 16)

            elif is_ext_within(img_fname, SUPPORT_COMMON_FORMAT):
                img = load_8bit_image(img_fname)
            else:
                self.logger.error(
                    f"Unsupported image format: {img_fname}. Only support"
                    f"{SUPPORT_COMMON_FORMAT + SUPPORT_RAW_FORMAT}.")
                return (img_fname, None)
        except (Exception, KeyboardInterrupt) as e:
            self.logger.error(
                f"Failed to load image: {img_fname} with error: {e.__repr__()}."
            )
        return (img_fname, img)


class ThreadImgLoader(VanillaImgLoader):

    def __init__(self,
                 img_fn_list: list[str],
                 raw_power: float = 2.222,
                 target_nl_mean: float = 0.3,
                 contrast_alpha: float = 1.2,
                 output_bps: int = 8,
                 maxsize: int = 1,
                 logger: BaseMetLog = get_default_logger(),
                 **kwargs: Any) -> None:
        super().__init__(img_fn_list, raw_power, target_nl_mean,
                         contrast_alpha, output_bps, **kwargs)
        self.queue: queue.Queue[ImgPair] = queue.Queue(maxsize=maxsize)
        self.stopped: bool = False

    def _load_images(self):
        try:
            while not self.stopped:
                fname, img = self._pop()
                self.queue.put((fname, img), timeout=IMG_GET_TIMEOUT)
                if fname is None or img is None:
                    break
        except Exception as e:
            self.logger.error(
                f"{self.__class__.__name__} is terminated due to {e.__repr__()}"
            )
        finally:
            self.stopped = True

    def pop(self):
        try:
            if not (self.stopped and self.queue.empty()):
                return self.queue.get(timeout=IMG_GET_TIMEOUT)
        except queue.Empty as e:
            return None, None
        return None, None

    def clear_queue(self):
        """clear queue.
        """
        while not self.queue.empty():
            self.queue.get()

    def start(self):
        self.clear_queue()
        self.stopped = False
        self.thread = threading.Thread(target=self._load_images)
        self.thread.setDaemon(True)
        self.thread.start()

    def stop(self):
        self.stopped = True
        self.thread.join()


class MultiThreadImgLoader(VanillaImgLoader):
    """Multi-threaded image loader with ordered output.

    Uses a pool of worker threads to load images in parallel but ensures
    `pop()` returns images in the original input order.
    """

    def __init__(self,
                 img_fn_list: list[str],
                 raw_power: float = 2.222,
                 target_nl_mean: float = 0.3,
                 contrast_alpha: float = 1.2,
                 output_bps: int = 8,
                 num_workers: int = DEFAULT_WORKER_NUM,
                 max_prefetch: Optional[int] = None,
                 logger: BaseMetLog = get_default_logger(),
                 **kwargs: Any) -> None:
        super().__init__(img_fn_list,
                         raw_power,
                         target_nl_mean,
                         contrast_alpha,
                         output_bps,
                         logger=logger,
                         **kwargs)
        self.num_workers = max(1, int(num_workers))
        # maximum number of images that can be loaded but not yet popped
        if max_prefetch is None:
            max_prefetch = max(2, self.num_workers * 2)
        self.max_prefetch = max(1, int(max_prefetch))

        # assignment counter: next index to be loaded by any worker
        self._next_assign_idx = 0
        self._assign_lock = threading.Lock()

        # results buffer: map from index -> (fname, img)
        self.results: dict[int, ImgPair] = {}
        self.results_lock = threading.Lock()
        self.results_cond = threading.Condition(self.results_lock)
        self.next_pop_idx = 0

        # semaphore to limit in-flight (loaded but not popped) images
        self._space_sem = threading.Semaphore(self.max_prefetch)

        self.workers: list[threading.Thread] = []
        self.stopped = False
        self._workers_alive = 0

    def _load_single(self, idx: int) -> ImgPair:
        """Load a single image by index using the same logic as VanillaImgLoader._pop.
        Returns (fname, img) or (None, None) on failure/end.
        """
        if idx < 0 or idx >= self.num_images:
            return (None, None)
        img_fname = self.img_fn_list[idx]
        try:
            if is_ext_within(img_fname, SUPPORT_RAW_FORMAT):
                img = load_raw_with_preprocess(
                    img_fname,
                    power=self.raw_power,
                    target_nl_mean=self.target_nl_mean,
                    contrast_alpha=self.contrast_alpha,
                    output_bps=8 if self.output_bps == 8 else 16)

            elif is_ext_within(img_fname, SUPPORT_COMMON_FORMAT):
                img = load_8bit_image(img_fname)
            else:
                self.logger.error(
                    f"Unsupported image format: {img_fname}. Only support"
                    f"{SUPPORT_COMMON_FORMAT + SUPPORT_RAW_FORMAT}.")
                return (img_fname, None)
        except (Exception, KeyboardInterrupt) as e:
            self.logger.error(
                f"Failed to load image: {img_fname} with error: {e.__repr__()}."
            )
            return (img_fname, None)
        return (img_fname, img)

    def _worker(self, id: int):
        try:
            while not self.stopped:
                # acquire space for one in-flight image; timeout so we can exit on stop
                acquired = self._space_sem.acquire(timeout=MT_HEART_TIME)
                if not acquired:
                    if self.stopped:
                        break
                    else:
                        continue

                # get next index to process
                with self._assign_lock:
                    idx = self._next_assign_idx
                    self._next_assign_idx += 1

                # if no more indices, release the slot and exit
                if idx >= self.num_images:
                    self._space_sem.release()
                    break

                res = self._load_single(idx)
                with self.results_cond:
                    self.results[idx] = res
                    self.results_cond.notify_all()
        except Exception as e:
            self.logger.error(
                f"{self.__class__.__name__} worker#{id} terminated due to {e.__repr__()}"
            )
        finally:
            with self.results_cond:
                self._workers_alive -= 1
                self.logger.info(f"worker#{id} task finished.")
                self.results_cond.notify_all()

    def start(self):
        # prepare
        with self.results_lock:
            self.results.clear()
            self.next_pop_idx = 0
        with self._assign_lock:
            self._next_assign_idx = 0
        # reset semaphore
        self._space_sem = threading.Semaphore(self.max_prefetch)
        self.stopped = False
        # start workers
        self.workers = []
        self._workers_alive = self.num_workers
        for i in range(self.num_workers):
            t = threading.Thread(target=self._worker,kwargs={'id': i})
            t.setDaemon(True)
            t.start()
            self.workers.append(t)

    def pop(self):
        # return next image in order; block until available or finished
        with self.results_cond:
            # loop and re-check
            while True:
                # if next result ready, return it
                if self.next_pop_idx in self.results:
                    fname, img = self.results.pop(self.next_pop_idx)
                    self.next_pop_idx += 1
                    # free one slot so workers can load another image
                    try:
                        self._space_sem.release()
                    except Exception:
                        pass
                    # avoid returning empty result
                    if fname is None and img is None:
                        continue
                    return (fname, img)
                # if no workers alive and no pending tasks, return end
                if self._workers_alive <= 0 and self._next_assign_idx >= self.num_images and not self.results:
                    return (None, None)
                # wait for a notification or timeout
                self.results_cond.wait(timeout=MT_HEART_TIME)
                

    def stop(self):
        # signal stop and join workers
        self.stopped = True
        # release semaphores to unblock any waiting workers
        for _ in range(self.num_workers):
            try:
                self._space_sem.release()
            except Exception:
                pass
        for t in self.workers:
            if t.is_alive():
                t.join()
