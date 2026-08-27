from fractions import Fraction
from types import SimpleNamespace

import numpy as np

from MetLib.videowrapper import PyAVVideoWrapper


class FakeFrame:

    def __init__(self, pts: int, value: int):
        self.pts = pts
        self.value = value

    def to_ndarray(self, format: str):
        assert format == "bgr24"
        return np.full((1, 1, 3), self.value, dtype=np.uint8)


class FakePacket:

    def __init__(self, frames):
        self.frames = frames

    def decode(self):
        return self.frames


class FakeContainer:

    def __init__(self, frames):
        self.frames = frames
        self.duration = None
        self.seek_args = None

    def seek(self, *args, **kwargs):
        self.seek_args = (args, kwargs)

    def demux(self, _video):
        yield FakePacket(self.frames)


def make_wrapper(frames, fps=10.0, num_frames=10):
    wrapper = object.__new__(PyAVVideoWrapper)
    wrapper.video = SimpleNamespace(
        time_base=Fraction(1, 1000),
        start_time=0,
        duration=num_frames * 100,
        frames=len(frames),
    )
    wrapper.container = FakeContainer(frames)
    wrapper._target_fps = fps
    wrapper._num_frames = num_frames
    wrapper.video_frame_cache = list(frames)
    wrapper._cur_frame_idx = 0
    wrapper._last_frame_data = None
    wrapper._last_frame_time_sec = None
    wrapper._eof = True
    return wrapper


def pixel_value(frame) -> int:
    return int(frame[0, 0, 0])


def test_read_holds_previous_frame_across_pts_gap():
    wrapper = make_wrapper([
        FakeFrame(0, 1),
        FakeFrame(500, 2),
    ], num_frames=6)

    values = []
    for _ in range(6):
        status, frame = wrapper.read()
        assert status
        values.append(pixel_value(frame))

    assert values == [1, 1, 1, 1, 1, 2]
    assert wrapper.get_video_pos() == 6
    assert wrapper.read() == (False, None)


def test_read_skips_pts_that_regress_beyond_jitter_tolerance():
    wrapper = make_wrapper([
        FakeFrame(0, 1),
        FakeFrame(100, 2),
        FakeFrame(20, 99),
        FakeFrame(200, 3),
    ], num_frames=3)

    values = []
    for _ in range(3):
        status, frame = wrapper.read()
        assert status
        values.append(pixel_value(frame))

    assert values == [1, 2, 3]
    assert wrapper._last_frame_time_sec == 0.2


def test_set_to_preserves_target_frame_for_next_read():
    frames = [FakeFrame(pts, index) for index, pts in enumerate(
        (0, 100, 200, 300))]
    wrapper = make_wrapper(frames, num_frames=4)
    wrapper.video_frame_cache = []
    wrapper._eof = False

    assert wrapper.set_to(2)
    assert wrapper.get_video_pos() == 2
    assert wrapper._last_frame_time_sec == 0.2
    assert wrapper.video_frame_cache[0].pts == 300

    status, frame = wrapper.read()
    assert status
    assert pixel_value(frame) == 2
    assert wrapper.get_video_pos() == 3


def test_target_fps_prefers_guessed_rate_over_gap_affected_average():
    wrapper = object.__new__(PyAVVideoWrapper)
    wrapper.video = SimpleNamespace(
        guessed_rate=Fraction(20, 1),
        base_rate=Fraction(20, 1),
        average_rate=Fraction(11420, 721),
    )

    assert wrapper._select_target_fps() == 20.0


def test_pts_conversion_is_relative_to_stream_start():
    wrapper = make_wrapper([], fps=10.0)
    wrapper.video.start_time = 100

    assert wrapper.pts2frame(200) == 1
    assert wrapper.frame2pts(1) == 200
