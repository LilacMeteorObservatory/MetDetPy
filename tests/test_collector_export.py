"""
Unit tests for the refactored Collector→Exporter export pipeline.
Tests COL-006 (waiting_meteor removal) and COL-023 (clip_merge_interval separation).
"""
import time
import threading
from unittest.mock import patch, MagicMock
from typing import Optional

import numpy as np

import sys
sys.path.insert(0, '.')

from MetLib.collector import MetExporter, Name2Label
from MetLib.metstruct import (MDTarget, ModelCfg, SingleMDRecord, RecheckCfg,
                              RuntimeParams)


def make_target(start_frame: int, last_activate_frame: int,
                category: str = "meteor",
                score: float = 0.8) -> MDTarget:
    """Helper to construct a minimal MDTarget for testing."""
    return MDTarget(
        start_frame=start_frame,
        start_time=f"00:00:{start_frame // 30:02d}.{(start_frame % 30) * 33:03d}",
        end_time=f"00:00:{last_activate_frame // 30:02d}.{(last_activate_frame % 30) * 33:03d}",
        last_activate_frame=last_activate_frame,
        last_activate_time=f"00:00:{last_activate_frame // 30:02d}.{(last_activate_frame % 30) * 33:03d}",
        duration=last_activate_frame - start_frame + 1,
        speed=5.0,
        dist=50.0,
        fix_dist=5.0,
        fix_speed=5.0,
        fix_motion_duration=0.5,
        fix_duration=0.5,
        num_pts=3,
        category=category,
        pt1=[100, 100],
        pt2=[200, 200],
        drct_loss=0.1,
        drct_cv=0.05,
        score=score,
        real_dist=50.0,
    )


def make_exporter(clip_merge_interval: float = 120.0,
                  recheck: bool = False) -> MetExporter:
    """Construct a MetExporter with mocked dependencies."""
    recheck_cfg = RecheckCfg(switch=recheck,
                             model=ModelCfg(name="mock", weight_path="mock.onnx",
                                           dtype="fp32", nms=True, warmup=False,
                                           pos_thre=0.5, nms_thre=0.45,
                                           multiscale_pred=1,
                                           multiscale_partition=1))
    runtime_param = MagicMock(spec=RuntimeParams)
    runtime_param.positive_category_list = ["meteor"]
    runtime_param.raw_size = [1920, 1080]
    runtime_param.runtime_size = [960, 540]
    runtime_param.fps = 30.0

    with patch('MetLib.collector.init_model'):
        exporter = MetExporter(
            recheck_cfg=recheck_cfg,
            runtime_param=runtime_param,
            video_loader=None,
            logger=MagicMock(),
            clip_merge_interval=clip_merge_interval,
            det_thre=0.7,
        )
    return exporter


def wait_for_processing(exporter: MetExporter, timeout: float = 2.0):
    """Send END_FLAG and wait for the export loop to finish."""
    exporter.export(exporter.END_FLAG, [], 99999, None)
    exporter.export_loop.join(timeout=timeout)


class TestFlushTiming:
    """Test that flush is triggered by cur_frame progression + gap detection."""

    def test_heartbeat_triggers_flush(self):
        """Target in pending + heartbeat with large cur_frame + nearest=None → flush."""
        exporter = make_exporter(clip_merge_interval=120.0)

        target = make_target(start_frame=10, last_activate_frame=40)
        exporter.export(exporter.ACTIVE_FLAG, [target], 50, None)
        # Heartbeat with cur_frame far enough: 40 + 120 < 200
        exporter.export(exporter.DROP_FLAG, [], 200, None)

        wait_for_processing(exporter)

        confirmed = [r for r in exporter.meteor_list
                     if r.target[0].category != "dropped"]
        assert len(confirmed) == 1
        assert confirmed[0].target[0].start_frame == 10

    def test_nearest_active_blocks_flush(self):
        """Pending target + heartbeat with nearest_active_start within interval → no flush until cleared."""
        exporter = make_exporter(clip_merge_interval=120.0)

        target = make_target(start_frame=10, last_activate_frame=40)
        exporter.export(exporter.ACTIVE_FLAG, [target], 50, None)
        # Heartbeat: cur_frame=200 exceeds gap, but nearest_active_start=100
        # 100 - 40 = 60 < 120 → should NOT flush yet
        exporter.export(exporter.DROP_FLAG, [], 200, 100)
        # Now clear: nearest=None, cur_frame=250
        exporter.export(exporter.DROP_FLAG, [], 250, None)

        wait_for_processing(exporter)

        confirmed = [r for r in exporter.meteor_list
                     if r.target[0].category != "dropped"]
        assert len(confirmed) == 1

    def test_nearest_active_far_does_not_block(self):
        """nearest_active_start far from pending → flush proceeds."""
        exporter = make_exporter(clip_merge_interval=120.0)

        target = make_target(start_frame=10, last_activate_frame=40)
        exporter.export(exporter.ACTIVE_FLAG, [target], 50, None)
        # nearest_active_start=300, far from pending[-1].last_activate=40
        # 300 - 40 = 260 > 120 → should flush
        exporter.export(exporter.DROP_FLAG, [], 200, 300)

        wait_for_processing(exporter)

        confirmed = [r for r in exporter.meteor_list
                     if r.target[0].category != "dropped"]
        assert len(confirmed) == 1


class TestClipMerging:
    """Test that temporally close targets are merged into one SingleMDRecord."""

    def test_two_targets_merge_into_one_clip(self):
        """Two targets within clip_merge_interval → 1 SingleMDRecord."""
        exporter = make_exporter(clip_merge_interval=120.0)

        t1 = make_target(start_frame=10, last_activate_frame=40)
        t2 = make_target(start_frame=80, last_activate_frame=110)
        # 80 < 40 + 120 → should merge
        exporter.export(exporter.ACTIVE_FLAG, [t1], 50, None)
        exporter.export(exporter.ACTIVE_FLAG, [t2], 120, None)
        # Trigger flush
        exporter.export(exporter.DROP_FLAG, [], 300, None)

        wait_for_processing(exporter)

        confirmed = [r for r in exporter.meteor_list
                     if r.target[0].category != "dropped"]
        assert len(confirmed) == 1
        assert len(confirmed[0].target) == 2

    def test_two_targets_separate_clips(self):
        """Two targets with gap > clip_merge_interval → 2 separate records."""
        exporter = make_exporter(clip_merge_interval=120.0)

        t1 = make_target(start_frame=10, last_activate_frame=40)
        t2 = make_target(start_frame=200, last_activate_frame=230)
        # 200 > 40 + 120 → separate
        exporter.export(exporter.ACTIVE_FLAG, [t1], 50, None)
        # First heartbeat triggers flush of t1 (200 - 40 = 160 > 120)
        exporter.export(exporter.ACTIVE_FLAG, [t2], 240, None)
        exporter.export(exporter.DROP_FLAG, [], 400, None)

        wait_for_processing(exporter)

        confirmed = [r for r in exporter.meteor_list
                     if r.target[0].category != "dropped"]
        assert len(confirmed) == 2


class TestRecheckIntegration:
    """Test that recheck rejections break the pending chain."""

    def test_fp_rejection_breaks_chain(self):
        """A-B-C targets, B rejected by recheck → A and C separate."""
        exporter = make_exporter(clip_merge_interval=120.0, recheck=True)

        t_a = make_target(start_frame=10, last_activate_frame=40)
        t_b = make_target(start_frame=80, last_activate_frame=110)
        t_c = make_target(start_frame=150, last_activate_frame=180)

        def mock_recheck(target):
            if target.start_frame == 80:
                return None  # reject B
            return target

        exporter.recheck_single_target = mock_recheck

        exporter.export(exporter.ACTIVE_FLAG, [t_a], 50, None)
        exporter.export(exporter.ACTIVE_FLAG, [t_b], 120, None)
        exporter.export(exporter.ACTIVE_FLAG, [t_c], 190, None)
        # Trigger flush: 400 - 180 = 220 > 120
        exporter.export(exporter.DROP_FLAG, [], 400, None)

        wait_for_processing(exporter)

        confirmed = [r for r in exporter.meteor_list
                     if r.target[0].category == "meteor"]
        # A(end=40) and C(start=150): 150 > 40+120 → separate
        assert len(confirmed) == 2
        assert confirmed[0].target[0].start_frame == 10
        assert confirmed[1].target[0].start_frame == 150

    def test_all_rejected_no_confirmed(self):
        """All targets rejected → no confirmed output (all become OTHERS/dropped)."""
        exporter = make_exporter(clip_merge_interval=120.0, recheck=True)

        targets = [make_target(start_frame=i * 30, last_activate_frame=i * 30 + 20)
                   for i in range(5)]

        exporter.recheck_single_target = lambda t: None

        for t in targets:
            exporter.export(exporter.ACTIVE_FLAG, [t], t.last_activate_frame + 10, None)
        exporter.export(exporter.DROP_FLAG, [], 500, None)

        wait_for_processing(exporter)

        # Rejected "meteor" targets get relabeled to OTHERS; none should keep "meteor"
        confirmed = [r for r in exporter.meteor_list
                     if r.target[0].category == "meteor"]
        assert len(confirmed) == 0
        assert exporter.logger.dropped.call_count == 5


class TestDropFlagWithData:
    """Test DROP_FLAG with non-empty data (label rewrite to 'dropped')."""

    def test_drop_flag_relabels_to_dropped(self):
        """DROP_FLAG targets get category rewritten to 'dropped' and logged."""
        exporter = make_exporter(clip_merge_interval=120.0)

        target = make_target(start_frame=10, last_activate_frame=40,
                             category="meteor")
        exporter.export(exporter.DROP_FLAG, [target], 50, None)

        wait_for_processing(exporter)

        assert len(exporter.meteor_list) == 1
        assert exporter.meteor_list[0].target[0].category == "DROPPED"
        exporter.logger.dropped.assert_called_once()

    def test_drop_flag_multiple_targets(self):
        """Multiple targets in a single DROP_FLAG message all get relabeled."""
        exporter = make_exporter(clip_merge_interval=120.0)

        targets = [make_target(start_frame=i * 30, last_activate_frame=i * 30 + 20)
                   for i in range(3)]
        exporter.export(exporter.DROP_FLAG, targets, 100, None)

        wait_for_processing(exporter)

        assert len(exporter.meteor_list) == 3
        for r in exporter.meteor_list:
            assert r.target[0].category == "DROPPED"
        assert exporter.logger.dropped.call_count == 3


class TestEndFlagFlush:
    """Test END_FLAG triggers immediate flush of pending_confirmed."""

    def test_end_flag_flushes_pending_without_gap(self):
        """END_FLAG flushes pending even when time gap hasn't been reached."""
        exporter = make_exporter(clip_merge_interval=120.0)

        target = make_target(start_frame=10, last_activate_frame=40)
        # Send target, then immediately END — no heartbeat with large cur_frame
        exporter.export(exporter.ACTIVE_FLAG, [target], 50, None)
        # cur_frame=60 is only 20 frames after last_activate=40, well below 120
        exporter.export(exporter.END_FLAG, [], 60, None)
        exporter.export_loop.join(timeout=2.0)

        confirmed = [r for r in exporter.meteor_list
                     if r.target[0].category != "dropped"]
        assert len(confirmed) == 1
        assert confirmed[0].target[0].start_frame == 10

    def test_end_flag_merges_close_pending(self):
        """Multiple close targets in pending are merged on END_FLAG."""
        exporter = make_exporter(clip_merge_interval=120.0)

        t1 = make_target(start_frame=10, last_activate_frame=40)
        t2 = make_target(start_frame=80, last_activate_frame=110)
        exporter.export(exporter.ACTIVE_FLAG, [t1], 50, None)
        exporter.export(exporter.ACTIVE_FLAG, [t2], 120, None)
        exporter.export(exporter.END_FLAG, [], 130, None)
        exporter.export_loop.join(timeout=2.0)

        confirmed = [r for r in exporter.meteor_list
                     if r.target[0].category != "dropped"]
        assert len(confirmed) == 1
        assert len(confirmed[0].target) == 2


class TestPendingSort:
    """Test that pending_confirmed is sorted by start_frame before flush."""

    def test_out_of_order_arrival_sorted_on_flush(self):
        """Targets arriving in non-start_frame order are sorted before merge."""
        exporter = make_exporter(clip_merge_interval=120.0)

        # Arrive out of order: start_frame 80 arrives before start_frame 10
        t_late = make_target(start_frame=80, last_activate_frame=110)
        t_early = make_target(start_frame=10, last_activate_frame=40)
        exporter.export(exporter.ACTIVE_FLAG, [t_late], 120, None)
        exporter.export(exporter.ACTIVE_FLAG, [t_early], 130, None)
        # Flush via heartbeat
        exporter.export(exporter.DROP_FLAG, [], 300, None)

        wait_for_processing(exporter)

        confirmed = [r for r in exporter.meteor_list
                     if r.target[0].category != "dropped"]
        # Both within interval (80 < 40+120), should merge into 1 record
        assert len(confirmed) == 1
        # First target in record should be the earlier one
        assert confirmed[0].target[0].start_frame == 10
        assert confirmed[0].target[1].start_frame == 80

    def test_out_of_order_separate_clips(self):
        """Out-of-order targets with gap > interval produce separate clips after sort."""
        exporter = make_exporter(clip_merge_interval=60.0)

        # Arrive out of order, but after sort the gap (200-40=160) > 60
        t_far = make_target(start_frame=200, last_activate_frame=230)
        t_near = make_target(start_frame=10, last_activate_frame=40)
        exporter.export(exporter.ACTIVE_FLAG, [t_far], 240, None)
        exporter.export(exporter.ACTIVE_FLAG, [t_near], 250, None)
        exporter.export(exporter.DROP_FLAG, [], 400, None)

        wait_for_processing(exporter)

        confirmed = [r for r in exporter.meteor_list
                     if r.target[0].category != "dropped"]
        assert len(confirmed) == 2
        # Sorted: first clip starts at 10, second at 200
        assert confirmed[0].target[0].start_frame == 10
        assert confirmed[1].target[0].start_frame == 200


class TestConfigCompat:
    """Test backward compatibility with configs missing clip_merge_interval."""

    def test_fallback_to_max_interval(self):
        """When clip_merge_interval is None in config, Collector uses max_interval."""
        from MetLib.metstruct import MeteorCfg
        cfg = MeteorCfg(
            min_len=15, max_interval=4, time_range=[0, 8],
            speed_range=[2, 21], drct_range=[0, 0.6],
            det_thre=0.7, thre2=2048
        )
        assert cfg.clip_merge_interval is None

    def test_explicit_clip_merge_interval(self):
        """When set explicitly, clip_merge_interval is used."""
        from MetLib.metstruct import MeteorCfg
        cfg = MeteorCfg(
            min_len=15, max_interval=4, time_range=[0, 8],
            speed_range=[2, 21], drct_range=[0, 0.6],
            det_thre=0.7, thre2=2048,
            clip_merge_interval=6.0
        )
        assert cfg.clip_merge_interval == 6.0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
