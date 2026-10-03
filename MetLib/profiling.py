"""Aggregate wall and selected per-thread CPU timings without per-frame storage."""

from contextlib import contextmanager, nullcontext
from functools import wraps
from threading import Lock
from time import perf_counter, thread_time


READ_CPU_STAGES = frozenset({
    "read.to_ndarray_bgr",
})


class StageProfiler:
    def __init__(self, name: str, cpu_stages=()):
        self.name = name
        self._cpu_stages = frozenset(cpu_stages)
        self._stats = {}
        self._lock = Lock()

    def call(self, stage, function, *args, **kwargs):
        start = perf_counter()
        cpu_start = thread_time() if stage in self._cpu_stages else None
        try:
            return function(*args, **kwargs)
        finally:
            cpu_elapsed = thread_time() - cpu_start if cpu_start is not None else None
            self._record(stage, perf_counter() - start, cpu_elapsed)

    @contextmanager
    def measure(self, stage):
        """Measure a block while leaving its ordinary calls visible."""
        start = perf_counter()
        cpu_start = thread_time() if stage in self._cpu_stages else None
        try:
            yield
        finally:
            cpu_elapsed = thread_time() - cpu_start if cpu_start is not None else None
            self._record(stage, perf_counter() - start, cpu_elapsed)

    def _record(self, stage, elapsed, cpu_elapsed=None):
        with self._lock:
            count, total, cpu_total = self._stats.get(stage, (0, 0.0, 0.0))
            self._stats[stage] = (count + 1, total + elapsed,
                                  cpu_total + (cpu_elapsed or 0.0))

    def summary(self):
        with self._lock:
            stats = sorted(self._stats.items())
        lines = []
        for stage, (count, total, cpu_total) in stats:
            line = (f"[Profile:{self.name}] {stage}: calls={count}; "
                    f"total={total:.6f}s; mean={total / count * 1000:.6f}ms")
            if stage in self._cpu_stages:
                ratio = f"{cpu_total / total * 100:.2f}%" if total > 0 else "n/a"
                line += (f"; thread_cpu_total={cpu_total:.6f}s; "
                         f"thread_cpu_mean={cpu_total / count * 1000:.6f}ms; "
                         f"thread_cpu_wall_ratio={ratio}")
            lines.append(line)
        return lines


def timed_call(owner, stage, function, *args, **kwargs):
    """Objects without an attached profiler keep their normal call behavior."""
    profiler = getattr(owner, "stage_profiler", None)
    if profiler is None:
        return function(*args, **kwargs)
    return profiler.call(stage, function, *args, **kwargs)


def profiled(stage):
    """Measure an instance method only when its owner has a profiler."""
    def decorate(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            profiler = getattr(self, "stage_profiler", None)
            if profiler is None:
                return method(self, *args, **kwargs)
            return profiler.call(stage, method, self, *args, **kwargs)
        return wrapper
    return decorate


def profile_stage(profiler, stage):
    """Return a timing scope, or a no-op scope when profiling is disabled."""
    return profiler.measure(stage) if profiler is not None else nullcontext()


def profile_loader(loader, name):
    """Attach after construction to exclude exposure estimation and warmup."""
    profiler = StageProfiler(name, cpu_stages=READ_CPU_STAGES)
    loader.stage_profiler = profiler
    loader.video.stage_profiler = profiler
    loader.preprocess.stage_profiler = profiler
    return profiler
