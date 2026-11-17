"""复核叠图的帧缓存：保留主路径 gray 之前的彩色帧，供序列增量叠图。

时序约束：

* 复核窗口是 ``[start_frame − padding, min(last_activate + padding, total − 1))``；
* 候选要等 ``max_interval`` 之后才确认结束，静默期**不能**进叠图；
* 静默期内 ``last_activate_frame`` 仍可能前进，而最大值归约不可撤销。

处理办法：
- 落在**当前窗口内**的帧在 loader 线程里立刻并入叠图；
- 落在窗口右端之后的帧先落进**分桶历史**（每 ``bucket_span``
  帧一个最大值桶），序列收到新响应时用"整桶合并"把新进入窗口的部分补上。
- 若桶与环形缓冲都覆盖不到，则记录待补读区间，由调用方用 ``max_stacker`` 补齐。
"""
import threading
from collections import deque
from typing import Optional

import numpy as np

from .utils import U8Mat


class _BucketHistory:
    """把"折叠前沿之后"的帧归约成固定跨度的最大值桶。

    桶边界锚定在构造时的折叠前沿（``limit``），之后不再移动：折叠区间的左端永远是
    ``max(folded_upto+1, limit)``，因此"包含左端的那个桶"可以直接整桶合并 —— 它多含的
    帧都落在 ``[limit, folded_upto]`` 内，既已折叠、又在窗口内，合并是幂等的。
    """

    def __init__(self, span: int, count: int, anchor: int) -> None:
        self.span = max(1, int(span))
        self.count = max(1, int(count))
        self.anchor = int(anchor)
        self._items: deque[tuple[int, U8Mat]] = deque()
        self.lost: list[tuple[int, int]] = []

    def __len__(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        self._items.clear()

    def _bucket_start(self, index: int) -> int:
        return self.anchor + (
            (int(index) - self.anchor) // self.span) * self.span

    def push(self, index: int, frame: U8Mat, keep_from: int) -> None:
        """把一帧并入它所属的桶；``keep_from`` 之前的桶被裁掉时记为缺失。"""
        start = self._bucket_start(index)
        if self._items:
            last_start, last_img = self._items[-1]
            if start == last_start:
                if last_img.shape == frame.shape:
                    np.maximum(last_img, frame, out=last_img)
                return
            if start < last_start:
                return  # 乱序/回退：忽略
        self._items.append((start, frame.copy()))
        self._trim(keep_from)

    def _trim(self, keep_from: int) -> None:
        while len(self._items) > self.count:
            start, _ = self._items[0]
            end = start + self.span - 1
            if end >= keep_from:
                # 桶深不足：这些帧已无处可取，如实记账（由调用方计入 refetch）
                self.lost.append((max(start, keep_from), end))
            self._items.popleft()

    def take(self, start: int, end: int) -> tuple[list[U8Mat], int]:
        """取出可与 ``[start, end]`` 合并的整桶，返回 (桶列表, 已覆盖到的帧号)。"""
        imgs: list[U8Mat] = []
        covered = start - 1
        while self._items:
            bucket_start, img = self._items[0]
            bucket_end = bucket_start + self.span - 1
            if bucket_end < start:
                self._items.popleft()  # 早于折叠前沿：已折叠过，安全丢弃
                continue
            if bucket_start <= start <= bucket_end and bucket_end <= end:
                imgs.append(img)  # 含左端的桶：整桶合并（多含的都是已折叠帧）
            elif bucket_start >= start and bucket_end <= end:
                imgs.append(img)
            else:
                break  # 右端不足一桶：交给环形缓冲逐帧补
            covered = max(covered, bucket_end)
            self._items.popleft()
        return imgs, covered


class SeriesStackBuilder:
    """单个序列的增量叠图（最大值归约）。

    窗口语义与旧路径 ``max_stacker(start=start−pad, end=L+pad)`` 保持一致：
    ``end`` 是**开区间**（``_batch_stacker`` 用 ``end−start`` 作为读取帧数），
    因此实际折叠的是 ``[start−pad, min(L+pad, total−1))``。

    Args:
        start_frame (int): 序列起始帧（runtime 帧号）。
        last_activate_frame (int): 当前最后一次响应帧。
        padding (int): 复核窗口两侧的补帧数（帧）。
        frame_count (Optional[int]): 视频总帧数，用于复刻旧路径的尾部夹取。
        bucket_span (Optional[int]): 分桶跨度（帧）。None 或 0 表示不启用分桶
            （退回"只有环形缓冲"的旧形态，便于单测与对照）。
        bucket_count (Optional[int]): 分桶个数上限，见 bucket_span。
    """

    def __init__(self,
                 start_frame: int,
                 last_activate_frame: int,
                 padding: int,
                 frame_count: Optional[int] = None,
                 bucket_span: Optional[int] = None,
                 bucket_count: Optional[int] = None) -> None:
        self.start_frame = int(start_frame)
        self.padding = int(padding)
        self.frame_count = frame_count
        self._limit = self._window_end(int(last_activate_frame))
        self._stack: Optional[U8Mat] = None
        # 窗口起点是 max(0, start−padding)，因此"已折叠到的帧"从它前面一帧开始
        self._window_start = max(0, self.start_frame - self.padding)
        self._folded_upto = self._window_start - 1
        self._newest_seen = self._window_start - 1
        self._lost_from: Optional[int] = None
        self._lost_to: Optional[int] = None
        self._closed = False
        span = int(bucket_span) if bucket_span else 0
        count = int(bucket_count) if bucket_count else 0
        self._buckets: Optional[_BucketHistory] = (_BucketHistory(
            span, count, self._limit) if span > 0 and count > 0 else None)
        # loader 线程 push 与主线程 advance/finish 会并发；状态必须串行化
        self._lock = threading.Lock()

    def _window_end(self, last_activate_frame: int) -> int:
        """窗口右端（开区间），与旧路径的 ``min(L+pad, total−1)`` 一致。"""
        end = int(last_activate_frame) + self.padding
        if self.frame_count is not None:
            end = min(end, self.frame_count - 1)
        return end

    @property
    def stack(self) -> Optional[U8Mat]:
        return self._stack

    @property
    def folded_upto(self) -> int:
        return self._folded_upto

    @property
    def bucketed(self) -> int:
        """当前持有的分桶数（0 表示未启用分桶）。"""
        return 0 if self._buckets is None else len(self._buckets)

    def push(self, index: int, frame: U8Mat) -> None:
        """接收一帧（索引为 runtime 帧号，按序到达）。

        只有落在当前窗口内的帧会被折叠；窗口右端之后的帧进分桶历史，
        留给 :meth:`advance` 合并。
        """
        with self._lock:
            self._push(index, frame)

    def _push(self, index: int, frame: U8Mat) -> None:
        if self._closed:
            return
        index = int(index)
        if index > self._newest_seen:
            self._newest_seen = index
        if index <= self._folded_upto:
            return
        if index >= self._limit:
            if self._buckets is not None:
                self._buckets.push(index, frame, self._folded_upto + 1)
                self._drain_bucket_lost()
            return
        if index > self._folded_upto + 1:
            # 交付出现跳跃（例如环被冲掉），记录缺失区间
            self._record_lost(self._folded_upto + 1, index - 1)
        self._fold(index, frame)

    def _merge(self, img: U8Mat) -> None:
        """整桶合并：不改变 folded_upto（桶可能多含已折叠帧）。"""
        if self._stack is None:
            self._stack = img.copy()
        elif self._stack.shape == img.shape:
            np.maximum(self._stack, img, out=self._stack)

    def _fold(self, index: int, frame: U8Mat) -> None:
        if self._stack is None:
            # 必须独占一份：调用方可能复用/改写原帧
            self._stack = frame.copy()
        elif self._stack.shape != frame.shape:
            # 形状不一致（分辨率切换/坏帧）时跳过，保留已有叠图
            return
        else:
            np.maximum(self._stack, frame, out=self._stack)
        if index > self._folded_upto:
            self._folded_upto = index

    def _record_lost(self, start: int, end: int) -> None:
        if end < start:
            return
        self._lost_from = (start if self._lost_from is None else min(
            self._lost_from, start))
        self._lost_to = (end if self._lost_to is None else max(
            self._lost_to, end))

    def _drain_bucket_lost(self) -> None:
        if self._buckets is None or not self._buckets.lost:
            return
        for start, end in self._buckets.lost:
            self._record_lost(start, end)
        self._buckets.lost.clear()

    def _consume(self, from_idx: int, to_idx: int,
                 frames: Optional[list[tuple[int, U8Mat]]]) -> None:
        """把 ``[from_idx, to_idx]`` 合并进叠图：先整桶、再逐帧（环形缓冲）。"""
        if to_idx < from_idx:
            return
        cursor = from_idx
        if self._buckets is not None:
            imgs, covered = self._buckets.take(from_idx, to_idx)
            for img in imgs:
                self._merge(img)
            if covered >= from_idx:
                self._folded_upto = max(self._folded_upto, covered)
                cursor = covered + 1
            self._drain_bucket_lost()
        supplied = {}
        if frames:
            supplied = {i: f for i, f in frames if cursor <= i <= to_idx}
        for i in range(cursor, to_idx + 1):
            frame = supplied.get(i)
            if frame is None:
                self._record_lost(i, i)
            else:
                self._fold(i, frame)

    def advance(self,
                last_activate_frame: int,
                frames: Optional[list[tuple[int, U8Mat]]] = None) -> None:
        """序列收到新响应：窗口右端后移，把新进入窗口的帧补叠进来。

        Args:
            last_activate_frame (int): 新的最后响应帧。
            frames (Optional[list[tuple[int, U8Mat]]]): 可用的历史帧（通常直接由
                帧缓存的环形缓冲提供）。缺的帧会被记为待补读区间。
        """
        with self._lock:
            self._advance(last_activate_frame, frames)

    def _advance(self, last_activate_frame: int,
                 frames: Optional[list[tuple[int, U8Mat]]]) -> None:
        new_limit = self._window_end(last_activate_frame)
        if new_limit <= self._limit:
            return
        self._consume(max(self._folded_upto + 1, self._limit),
                      min(new_limit - 1, self._newest_seen), frames)
        self._limit = new_limit
        # 只对"已经看到过"的帧判断缺失
        want_upto = min(new_limit - 1, self._newest_seen)
        if self._folded_upto < want_upto:
            self._record_lost(self._folded_upto + 1, want_upto)

    def close(self) -> None:
        """标记封口：之后到达的帧不再进入叠图。"""
        with self._lock:
            self._closed = True

    def finish(
        self,
        marked: Optional[int] = None,
        frames: Optional[list[tuple[int, U8Mat]]] = None
    ) -> tuple[Optional[U8Mat], Optional[tuple[int, int]]]:
        """封口：返回 (叠图, 需要补读的帧区间)。

        窗口右端之后的帧（静默期）不进入叠图；若增量叠图缺帧，
        返回闭区间 ``[from, to]`` 供调用方用 ``max_stacker`` 补齐。

        Args:
            marked (Optional[int]): 交付到的最大帧号（省略时用本构建器见过的最大值）。
            frames (Optional[list[tuple[int, U8Mat]]]): 封口时还能用的历史帧，用于
                补最后一段（通常是环形缓冲的快照）。
        """
        with self._lock:
            self._closed = True
            want_upto = min(
                self._limit - 1,
                int(marked) if marked is not None else self._newest_seen)
            if self._folded_upto < want_upto:
                if self._buckets is not None or frames:
                    self._consume(self._folded_upto + 1, want_upto, frames)
                if self._folded_upto < want_upto:
                    self._record_lost(self._folded_upto + 1, want_upto)
            if self._lost_from is None or self._lost_to is None:
                return self._stack, None
            to = min(self._lost_to, self._limit - 1)
            if to < self._lost_from:
                return self._stack, None
            return self._stack, (self._lost_from, to)


class RecheckFrameCache:
    """线程安全的双路输出落点：主 loader 每帧推入，序列按需叠图。

    Args:
        padding (int): 复核窗口补帧数（帧）。
        extra_capacity (int): 环形缓冲在 ``padding`` 之外的额外帧数，需要覆盖
            "loader 领先主线程的帧数 + 分桶边界 + 余量"（见模块文档的预算公式）。
        bucket_span (Optional[int]): 分桶跨度，None/0 表示不启用分桶。
        bucket_count (Optional[int]): 分桶个数上限。
    """

    def __init__(self,
                 padding: int = 1,
                 extra_capacity: int = 8,
                 bucket_span: Optional[int] = None,
                 bucket_count: Optional[int] = None) -> None:
        self.padding = max(0, int(padding))
        self.extra_capacity = max(1, int(extra_capacity))
        self.bucket_span = int(bucket_span) if bucket_span else 0
        self.bucket_count = int(bucket_count) if bucket_count else 0
        self.frame_count: Optional[int] = None
        self._ring: deque[tuple[int, U8Mat]] = deque(maxlen=self.padding +
                                                     self.extra_capacity)
        self._builders: dict[int, SeriesStackBuilder] = {}
        self._lock = threading.Lock()
        self._last_index = -1
        self.frame_count_seen = 0

    @property
    def marked(self) -> int:
        """当前交付到的最大帧号（= loader 已推入的最后一帧）。"""
        return self._last_index

    @property
    def payload_frames(self) -> int:
        """环形缓冲当前持有的帧数。"""
        return len(self._ring)

    @property
    def capacity(self) -> int:
        """环形缓冲的容量（帧）。"""
        return self._ring.maxlen or 0

    def describe(self) -> str:
        """一行预算描述，用于运行日志核对。"""
        return (f'pad={self.padding}, ring={self.capacity}, '
                f'buckets={self.bucket_count}x{self.bucket_span}')

    def configure(self,
                  padding: int,
                  frame_count: Optional[int] = None,
                  extra_capacity: Optional[int] = None,
                  bucket_span: Optional[int] = None,
                  bucket_count: Optional[int] = None) -> None:
        """在知道帧率/总帧数/loader 队列深度之后重设预卷长度与分桶。"""
        with self._lock:
            self.padding = max(0, int(padding))
            self.frame_count = frame_count
            if extra_capacity is not None:
                self.extra_capacity = max(1, int(extra_capacity))
            if bucket_span is not None:
                self.bucket_span = max(0, int(bucket_span))
            if bucket_count is not None:
                self.bucket_count = max(0, int(bucket_count))
            capacity = self.padding + self.extra_capacity
            if self._ring.maxlen != capacity:
                self._ring = deque(self._ring, maxlen=capacity)

    def reset(self) -> None:
        """重置位置状态（正式读取开始前调用，用于清掉曝光估计阶段的重读）。"""
        with self._lock:
            self._ring.clear()
            self._last_index = -1
            self.frame_count_seen = 0
            for builder in self._builders.values():
                builder.close()
            self._builders.clear()

    def push(self, index: int, frame: U8Mat) -> None:
        """由主 loader 在每帧预处理时调用（loader 线程）。

        入环的同时，把落在各序列当前窗口内的帧立即叠进去 ——
        折叠成本留在 loader 线程，不占用主循环。
        """
        with self._lock:
            if index <= self._last_index:
                # 反向/重复读取（例如曝光估计阶段的重读）不计入
                return
            self._last_index = index
            self._ring.append((index, frame))
            self.frame_count_seen += 1
            builders = list(self._builders.values())
        for builder in builders:
            builder.push(index, frame)

    def snapshot(self) -> list[tuple[int, U8Mat]]:
        """当前环形缓冲的快照（供封口时补最后一帧段用）。"""
        with self._lock:
            return list(self._ring)

    def attach(self, key: int, start_frame: int,
               last_activate_frame: int) -> SeriesStackBuilder:
        """为序列注册叠图，并用环形缓冲回填预卷帧。

        注册与预卷回填必须原子完成：否则可能存在异步问题。
        """
        with self._lock:
            builder = SeriesStackBuilder(start_frame, last_activate_frame,
                                         self.padding, self.frame_count,
                                         self.bucket_span, self.bucket_count)
            first = max(0, start_frame - self.padding)
            batch = [(i, f) for i, f in self._ring if i >= first]
            self._builders[key] = builder
            for index, frame_img in batch:
                builder.push(index, frame_img)
        return builder

    def advance(self, key: int, last_activate_frame: int) -> None:
        """序列有新响应：窗口右端后移，用分桶与环形缓冲里的历史帧补齐。"""
        with self._lock:
            builder = self._builders.get(key)
            if builder is None:
                return
            snapshot = list(self._ring)
        builder.advance(last_activate_frame, frames=snapshot)

    def detach(self, key: int) -> Optional[SeriesStackBuilder]:
        """注销序列并返回其叠图构建器（由调用方封口）。"""
        with self._lock:
            return self._builders.pop(key, None)

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._builders)
