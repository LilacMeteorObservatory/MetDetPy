"""Bounded, ordered image inference with one model per device worker."""
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from time import perf_counter
from typing import Optional, Callable

from .metlog import BaseMetLog
from .model import DEVICE_MAPPING, YOLOModel
from .onnx_devices import describe_adapter, discover_dml_adapters, same_adapter, parse_photo_devices


def create_photo_models(keys: list[str],
                        factory: Callable[[str, int], YOLOModel],
                        available: list[str],
                        logger: BaseMetLog,
                        num_threads: Optional[int] = None):
    """Initialize one representative per physical device, keeping bare DML automatic."""
    keys = parse_photo_devices(",".join(keys))
    automatic = keys[0] == "gpu"
    adapters = []
    # 指定gpu或dml设备，打印可用选项
    if automatic or any(key.partition(":")[0] == "dml" for key in keys):
        try:
            adapters = discover_dml_adapters()
            for adapter in adapters:
                logger.info(describe_adapter(adapter))
        except OSError as error:
            logger.warning(f"DML adapter discovery failed: {error}")

    groups = []
    # 构造候选provider_key
    if automatic and "dml" in available:
        for adapter in sorted(adapters, key=lambda a: a.index):
            if adapter.software:
                continue
            group = next((g for g in groups if same_adapter(g[0], adapter)),
                         None)
            if group is None:
                groups.append([adapter])
            else:
                group.append(adapter)
        candidates = [[group[0].key] for group in groups]
    elif automatic:
        logger.info(
            "DML unavailable; using existing single-device selection (no all-GPU enumeration)."
        )
        candidates = [["default"]]
    else:
        candidates = []
        selected = []
        for key in keys:
            index = int(key.split(":")[1]) if key.startswith("dml:") else None
            adapter = next(
                (a for a in adapters
                 if a.index == index), None) if index is not None else None
            if adapter is not None and adapter.software:
                raise ValueError(f"{key} is a software adapter, not a GPU.")
            if adapter is not None and any(
                    same_adapter(a, adapter) for a in selected):
                logger.info(f"Skipping duplicate physical device {key}.")
                continue
            if adapter is not None:
                selected.append(adapter)
            canonical = (f"dml:{adapter.duplicate_of}" if adapter is not None
                         and adapter.duplicate_of is not None else key)
            if any(canonical in group for group in candidates):
                continue
            candidates.append([canonical])

    models: list[tuple[str, YOLOModel]] = []

    def add(key: str, cpu_threads: int):
        model = factory(key, cpu_threads)
        actual = getattr(model.backend, "device", None)
        expected = DEVICE_MAPPING.get(key.partition(":")[0], [])
        if key != "default" and isinstance(
                actual,
                str) and expected and not actual.startswith(expected[0]):
            raise RuntimeError(f"Requested {key}, but initialized {actual}.")
        # An automatic non-DML default may already have selected CPU.
        if isinstance(actual,
                      str) and actual.startswith("CPUExecutionProvider"):
            if any(
                    isinstance(getattr(m.backend, "device", None), str)
                    and m.backend.device.startswith("CPUExecutionProvider")
                    for _, m in models):
                return
        models.append((key, model))
        logger.info(f"Photo inference device enabled: {key}.")

    for group in candidates:
        for key in group:
            threads = (num_threads if num_threads is not None else
                       1 if len(candidates) > 1 else 0) if key == "cpu" else (
                           num_threads if key == "default"
                           and num_threads is not None else 0)
            try:
                add(key, threads)
                break
            except Exception as error:
                logger.error(
                    f"Failed to initialize photo device {key}: {error!r}")
                if not automatic:
                    raise
    if automatic:
        if not models:
            logger.warning(
                "No usable automatic GPU session; falling back to CPU.")
            add("cpu", num_threads if num_threads is not None else 0)
        elif "cpu" in keys:
            # Avoid constructing an extra CPU session when default selected CPU.
            if not any(
                    isinstance(getattr(m.backend, "device", None), str)
                    and m.backend.device.startswith("CPUExecutionProvider")
                    for _, m in models):
                add("cpu", num_threads if num_threads is not None else 1)
    return models


@dataclass
class PhotoTask:
    index: int
    source: str
    image: object


@dataclass
class PhotoPrediction:
    task: PhotoTask
    image: object
    boxes: object
    scores: object


class PhotoInferencePool:
    """A shared queue dynamically feeds devices; output keeps source order.

    Only the consumer drives the source. At most 2 * device_count tasks, including
    completed results waiting for earlier tasks, are retained by the scheduler.
    """

    def __init__(self,
                 models: list[tuple[str, YOLOModel]],
                 prepare=None,
                 logger=None):
        if not models:
            raise ValueError("At least one inference model is required.")
        self.models = models
        self.prepare = prepare or (lambda image: image)
        self.logger = logger
        self.max_in_flight = 2 * len(models)
        self.queue = Queue(maxsize=self.max_in_flight)
        self.stats = {key: [0, 0.0] for key, _ in models}
        self.threads = []
        self.pending = deque()
        self.closed = False

    def _worker(self, key: str, model: YOLOModel):
        while True:
            item = self.queue.get()
            if item is None:
                return
            task, future = item
            if not future.set_running_or_notify_cancel():
                continue
            try:
                image = self.prepare(task.image)
                start = perf_counter()
                try:
                    boxes, scores = model.forward(image)
                finally:
                    self.stats[key][0] += 1
                    self.stats[key][1] += perf_counter() - start
                future.set_result(PhotoPrediction(task, image, boxes, scores))
            except Exception as error:
                message = f"Photo inference failed on {key}, input {task.source}: {error!r}"
                if self.logger is not None:
                    self.logger.error(message)
                future.set_exception(RuntimeError(message))
            finally:
                # An idle worker must not retain its last full-resolution image.
                item = task = future = None
                image = boxes = scores = None

    def __enter__(self):
        for key, model in self.models:
            thread = Thread(target=self._worker,
                            args=(key, model),
                            name=f"photo-{key}")
            thread.start()
            self.threads.append(thread)
        return self

    def map(self, tasks):
        source = iter(tasks)
        exhausted = False
        try:
            while True:
                while not exhausted and len(self.pending) < self.max_in_flight:
                    try:
                        task = next(source)
                    except StopIteration:
                        exhausted = True
                        break
                    future = Future()
                    self.pending.append(future)
                    self.queue.put((task, future))
                if not self.pending:
                    break
                future = self.pending[0]
                prediction = future.result()
                # Keep the delivered image inside the bound until consumed.
                yield prediction
                self.pending.popleft()
        finally:
            close = getattr(source, "close", None)
            if close is not None:
                close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        for future in self.pending:
            future.cancel()
        for _ in self.threads:
            self.queue.put(None)
        for thread in self.threads:
            thread.join()
        self.pending.clear()
        # Release queued task/image references, including cancelled work.
        while not self.queue.empty():
            self.queue.get_nowait()

    def summary(self):
        return [
            f"[Photo:{key}] calls={count}; total={total:.6f}s; "
            f"mean={total / count * 1000 if count else 0:.6f}ms"
            for key, (count, total) in self.stats.items()
        ]

    def __exit__(self, *args):
        self.close()
