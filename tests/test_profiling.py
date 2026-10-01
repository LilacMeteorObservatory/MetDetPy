import pytest
import numpy as np
from unittest.mock import MagicMock

import MetLib.profiling as profiling
from MetLib.Detector import M3Detector
from MetLib.metstruct import MainDetectCfg
from tests.test_videowrapper import FakeFrame, make_wrapper


def test_m3_profiled_window_preserves_detection_results():
    cfg = MainDetectCfg.from_json_file("config/m3det_normal.json")
    mask = np.ones((48, 64), dtype=np.uint8)
    plain = M3Detector(1, 3, mask, 4, cfg.detector.cfg, MagicMock())
    measured = M3Detector(1, 3, mask, 4, cfg.detector.cfg, MagicMock())
    profiler = profiling.StageProfiler("main_loop")
    measured.stack.stage_profiler = profiler
    frames = np.random.default_rng(0).integers(0, 30, (5, 48, 64), dtype=np.uint8)

    for frame in frames:
        plain.update(frame)
        measured.update(frame)
        plain_lines, plain_classes = plain.detect()
        measured_lines, measured_classes = measured.detect()
        np.testing.assert_array_equal(measured_lines, plain_lines)
        np.testing.assert_array_equal(measured_classes, plain_classes)

    summary = profiler.summary()
    assert len(summary) == 1
    assert "m3.update.stack: calls=5" in summary[0]


@pytest.mark.parametrize("use_scope", [False, True])
def test_cpu_timing_accumulates_success_and_failure(monkeypatch, use_scope):
    wall = iter([0.0, 4.0, 10.0, 12.0])
    cpu = iter([1.0, 2.0, 3.0, 3.5])
    monkeypatch.setattr(profiling, "perf_counter", lambda: next(wall))
    monkeypatch.setattr(profiling, "thread_time", lambda: next(cpu))
    profiler = profiling.StageProfiler("test", cpu_stages={"conversion"})

    def invoke(function):
        if use_scope:
            with profiling.profile_stage(profiler, "conversion"):
                return function()
        return profiler.call("conversion", function)

    assert invoke(lambda: 42) == 42

    def fail():
        raise ValueError("failed")

    with pytest.raises(ValueError, match="failed"):
        invoke(fail)
    assert profiler.summary() == [
        "[Profile:test] conversion: calls=2; total=6.000000s; mean=3000.000000ms; "
        "thread_cpu_total=1.500000s; thread_cpu_mean=750.000000ms; "
        "thread_cpu_wall_ratio=25.00%"
    ]


def test_disabled_scope_runs_normally_without_reading_clocks(monkeypatch):
    monkeypatch.setattr(profiling, "perf_counter", lambda: pytest.fail("clock read"))
    with profiling.profile_stage(None, "detect"):
        value = 42
    assert value == 42


def test_unselected_stage_does_not_read_cpu_clock(monkeypatch):
    def unexpected_clock():
        pytest.fail("CPU clock should only be read for selected stages")

    monkeypatch.setattr(profiling, "thread_time", unexpected_clock)
    profiler = profiling.StageProfiler("test", cpu_stages={"conversion"})
    assert profiler.call("other", lambda: 42) == 42
    assert "thread_cpu" not in profiler.summary()[0]


def test_loader_enables_cpu_timing_for_selected_stages():
    from types import SimpleNamespace

    loader = SimpleNamespace(video=SimpleNamespace(), preprocess=SimpleNamespace())
    profiler = profiling.profile_loader(loader, "main_loader")
    assert loader.video.stage_profiler is profiler
    assert loader.preprocess.stage_profiler is profiler
    for stage in profiling.READ_CPU_STAGES:
        assert profiling.timed_call(loader, stage, lambda: 42) == 42
    assert len(profiler.summary()) == len(profiling.READ_CPU_STAGES)
    assert all("thread_cpu_total=" in line for line in profiler.summary())


def test_timing_preserves_results_and_records_failed_attempts(monkeypatch):
    ticks = iter([1.0, 1.25, 2.0, 2.5])
    monkeypatch.setattr(profiling, "perf_counter", lambda: next(ticks))
    profiler = profiling.StageProfiler("test")
    assert profiler.call("read", lambda: 42) == 42

    def fail():
        raise ValueError("read failed")

    with pytest.raises(ValueError, match="read failed"):
        profiler.call("read", fail)
    assert profiler.summary() == [
        "[Profile:test] read: calls=2; total=0.750000s; mean=375.000000ms"
    ]


def test_decorator_keeps_plain_calls_and_records_failures(monkeypatch):
    ticks = iter([1.0, 1.25, 2.0, 2.5])
    monkeypatch.setattr(profiling, "perf_counter", lambda: next(ticks))

    class Worker:
        @profiling.profiled("work")
        def run(self, fail=False):
            if fail:
                raise ValueError("failed")
            return 42

    worker = Worker()
    assert worker.run() == 42
    worker.stage_profiler = profiling.StageProfiler("test")
    assert worker.run() == 42
    with pytest.raises(ValueError, match="failed"):
        worker.run(fail=True)
    assert worker.stage_profiler.summary() == [
        "[Profile:test] work: calls=2; total=0.750000s; mean=375.000000ms"
    ]


def test_demux_timing_excludes_time_between_generator_yields(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(profiling, "perf_counter", lambda: now[0])
    frame = FakeFrame(0, 7)
    wrapper = make_wrapper([frame])
    wrapper.stage_profiler = profiling.StageProfiler("loader")
    batches = wrapper._decoded_frame_batches()
    assert next(batches) == [frame]
    now[0] = 100.0  # Downstream processing must not count as demux time.
    with pytest.raises(StopIteration):
        next(batches)
    assert wrapper.stage_profiler.summary() == [
        "[Profile:loader] read.packet_decode: calls=1; total=0.000000s; mean=0.000000ms",
    ]


def test_profiled_read_preserves_frame_holding():
    wrapper = make_wrapper([FakeFrame(0, 7), FakeFrame(300, 9)], num_frames=5)
    wrapper.stage_profiler = profiling.StageProfiler("loader")
    values = [int(wrapper.read()[1][0, 0, 0]) for _ in range(5)]
    assert values == [7, 7, 7, 9, 9]
    # Held CFR frames reuse the array instead of converting it again.
    assert any("read.to_ndarray_bgr: calls=2;" in line
               for line in wrapper.stage_profiler.summary())


def test_read_profiles_conversion_without_changing_frame_selection(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(profiling, "perf_counter", lambda: now[0])

    class TimedFrame(FakeFrame):
        def to_ndarray(self, format="bgr24"):
            now[0] += 5.0
            return super().to_ndarray(format)

        def __del__(self):
            now[0] += 7.0

    def consume(_target):
        now[0] += 3.0
        return TimedFrame(0, 7)

    wrapper = make_wrapper([])
    wrapper._consume_until = consume
    wrapper.stage_profiler = profiling.StageProfiler("loader")
    status, frame = wrapper.read()
    assert status and int(frame[0, 0, 0]) == 7
    assert wrapper.stage_profiler.summary() == [
        "[Profile:loader] read.to_ndarray_bgr: calls=1; total=5.000000s; mean=5000.000000ms",
    ]


def test_loader_read_includes_frame_handoff(monkeypatch):
    from types import SimpleNamespace
    from MetLib.videoloader import VanillaVideoLoader

    now = [0.0]
    monkeypatch.setattr(profiling, "perf_counter", lambda: now[0])

    class OldFrame:
        def __del__(self):
            now[0] += 2.0

    loader = object.__new__(VanillaVideoLoader)
    loader.cur_frame = OldFrame()
    new_frame = object()
    loader.video = SimpleNamespace(read=lambda: (True, new_frame))
    loader.stage_profiler = profiling.StageProfiler("loader")
    assert loader._read_frame()
    assert loader.cur_frame is new_frame
    assert loader.stage_profiler.summary() == [
        "[Profile:loader] read.total: calls=1; total=2.000000s; mean=2000.000000ms",
    ]
